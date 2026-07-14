//! Shared REML/LAML contract types.
//!
//! These are the family-facing interfaces for REML outer assembly. They live
//! below `solver` so families can construct operator-backed derivative payloads
//! without importing `solver::estimate::reml::reml_outer_engine`.

use std::any::Any;
use std::collections::HashMap;
use std::panic::{AssertUnwindSafe, catch_unwind, resume_unwind};
use std::sync::{Arc, Condvar, Mutex};

use ndarray::{Array1, Array2, ArrayView1, ArrayView2, ArrayViewMut1, ArrayViewMut2};
use rayon::iter::{IntoParallelIterator, ParallelIterator};

#[macro_use]
mod macros;

pub mod basis_error;
pub mod block_count_error;
pub mod block_role;
pub mod block_spec;
pub mod coefficient_prior_mean;
pub mod custom_family_blockwise;
pub mod custom_family_error;
pub mod diagnostics;
pub mod dispersion;
pub mod dispersion_cov;
pub mod estimation_error;
pub mod execution_path;
pub mod family_options;
pub mod finite_validation;
pub mod fisher_rao;
pub mod gauge;
pub mod identifiability_audit;
pub mod joint_penalty;
mod constraint_set;
mod linear_constraints;
pub mod log_strength;
pub mod monotone_root_error;
pub mod outer_subsample;
pub mod penalty_coordinate;
pub mod penalty_matrix;
mod pseudo_logdet;
pub mod psi_design_contract;
pub mod psi_terms;
pub mod riemannian_retraction;
// `ρ`-posterior certificate/escalation DATA types contract-downed (#1521) so
// gam-solve can store/return them without a back-edge into gam-inference; the
// computation stays UP in the monolith `inference::rho_posterior`.
pub mod rho_posterior;
pub mod row_measure;
pub mod row_metric;
pub mod schedule;
// #1521 contract-downs: pure-data carriers + caller-supplied sampler/verdict
// traits so gam-solve can call up-tier work (NUTS sampling, topology verdicts)
// without a back-edge into gam-inference/gam-sae; computation stays UP.
pub mod laplace_sampler_contract;
mod seeding;
pub mod solver_contract;
pub mod topology_certificates;
pub mod types;

pub use riemannian_retraction::LatentRetractionRegistry;
pub use row_measure::RowSubsampleMask;

pub use basis_error::BasisError;
pub use block_count_error::BlockCountMismatch;
pub use block_role::BlockRole;
pub use block_spec::{
    AdditiveBlockJacobian, BlockEffectiveJacobian, BlockGeometryDirectionalDerivative,
    BlockWorkingSet, FamilyChannelHessian, FamilyLinearizationState, GaugeComposedJacobian,
    ParameterBlockSpec, ParameterBlockState, RowScaledJacobian,
};
pub use coefficient_prior_mean::{CoefficientPriorMean, PriorMeanError};
pub use custom_family_blockwise::{
    CUSTOM_FAMILY_RIDGE_FLOOR, ExactNewtonOuterCurvature, validate_blockspec_consistency,
};
pub use custom_family_error::CustomFamilyError;
pub use dispersion::{Dispersion, DispersionError};
pub use dispersion_cov::{
    CovarianceStandardErrorError, PhiScaledCovariance, UnscaledPrecision, se_from_covariance,
};
pub use estimation_error::{
    EstimationError, FixedLambdaCheckpoint, FixedLambdaResidualKind, FixedLambdaSolverStage,
    FixedLambdaStallReason, FixedLambdaStationarityEvidence,
};
pub use execution_path::ExecutionPath;
pub use family_options::{ExactNewtonOuterObjective, ExactOuterDerivativeOrder};
pub use finite_validation::{
    bail_if_cached_beta_non_finite, ensure_finite_scalar, ensure_finite_scalar_estimation,
    validate_all_finite, validate_all_finite_estimation,
};
pub use fisher_rao::{
    FisherRaoDefiniteness, normalize_fisher_rao_blocks, normalize_fisher_rao_blocks_pd,
};
use gam_linalg::dense;
pub use gam_linalg::faer_ndarray::{in_nested_parallel_region, with_nested_parallel};
pub use gauge::Gauge;
pub use identifiability_audit::{
    AliasedPair, BlockIdentity, DroppedColumn, IdentifiabilityAudit, MapUniquenessError,
};
pub use joint_penalty::{JointPenaltyBundle, JointPenaltyError, JointPenaltySpec};
pub use constraint_set::{ConstraintSet, KhatriRaoConeConstraints, PlacedConstraintBlock};
pub use linear_constraints::LinearInequalityConstraints;
pub use log_strength::{
    IndexedLogStrengthDomainError, LOG_STRENGTH_MAX, LOG_STRENGTH_MIN, LogStrengthDomainError,
    PhysicalStrengthDomainError, checked_exp_log_strength, checked_exp_log_strengths,
    checked_log_strength, validate_log_strength, validate_log_strengths,
};
pub use monotone_root_error::MonotoneRootError;
pub use penalty_coordinate::PenaltyCoordinate;
pub use penalty_matrix::PenaltyMatrix;
pub use pseudo_logdet::PseudoLogdetMode;
pub use psi_design_contract::{
    CustomFamilyBlockPsiDerivative, CustomFamilyHyperAxis, CustomFamilyHyperLayout,
    CustomFamilyPsiDerivativeOperator, JointHessianSourcePreference,
    MaterializablePsiDerivativeOperator, MaterializationIntent, SharedCustomFamilyHyperLayout,
};
pub use psi_terms::{
    ExactNewtonJointPsiSecondOrderContracted, ExactNewtonJointPsiSecondOrderTerms,
    ExactNewtonJointPsiTerms, ExactNewtonJointPsiWorkspace,
};
pub use row_metric::{
    FisherFactorKind, MetricProvenance, RowMetric, WeightField, pack_probe_factors,
};
pub use schedule::{GumbelTemperatureSchedule, ScheduleKind, SearchStrategy};
pub use seeding::{SeedConfig, SeedRiskProfile, clamp_seed_rho_to_bounds, normalize_seed_bounds};
pub use solver_contract::{
    DeclaredHessianForm, Derivative, EfsEval, FixedPointCertificateEval,
    FixedPointCoordinateCertificate, HessianMaterialization, HessianOperator, HessianValue,
    ObjectiveEvalError, OuterEval, OuterStrategyError,
};
pub use types::*;

#[cold]
fn reml_contract_panic(message: impl Into<String>) -> ! {
    std::panic::panic_any(message.into())
}

/// Evaluation mode for the unified evaluator.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum EvalMode {
    /// Compute cost only (e.g., for line search).
    ValueOnly,
    /// Compute cost and gradient (the common case).
    ValueAndGradient,
    /// Compute cost, gradient, and outer Hessian.
    ValueGradientHessian,
}

/// Trait for operators that can compute a hyper-derivative matrix-vector product
/// without necessarily materializing the full matrix.
struct NonDowncastableHyperOperator;

static NON_DOWNCASTABLE_HYPER_OPERATOR: NonDowncastableHyperOperator = NonDowncastableHyperOperator;

pub trait HyperOperator: Send + Sync {
    /// Operator dimension `p` such that `B · v` consumes a `p`-vector and
    /// produces a `p`-vector.
    fn dim(&self) -> usize;

    /// Compute B · v (matrix-vector product). v and result are p-vectors.
    fn mul_vec(&self, v: &Array1<f64>) -> Array1<f64>;

    /// Expose the concrete type for solver-local downcast helpers when the
    /// implementor has a `'static` concrete type. Borrowing adapters may keep
    /// the default, which simply cannot downcast.
    fn as_any(&self) -> &(dyn Any + 'static) {
        &NON_DOWNCASTABLE_HYPER_OPERATOR
    }

    /// Compute B · v from a vector view.
    fn mul_vec_view(&self, v: ArrayView1<'_, f64>) -> Array1<f64> {
        self.mul_vec(&v.to_owned())
    }

    /// Compute B · v into caller-owned storage.
    fn mul_vec_into(&self, v: ArrayView1<'_, f64>, mut out: ArrayViewMut1<'_, f64>) {
        out.assign(&self.mul_vec_view(v));
    }

    /// Compute B · F where F is (p × k). Default dispatches per-column in
    /// parallel unless already inside a rayon worker.
    fn mul_mat(&self, factor: &Array2<f64>) -> Array2<f64> {
        let p = factor.nrows();
        let k = factor.ncols();
        let mut out = Array2::<f64>::zeros((p, k));
        if rayon::current_thread_index().is_some() {
            for col in 0..k {
                let bv = out.column_mut(col);
                self.mul_vec_into(factor.column(col), bv);
            }
            return out;
        }
        let cols: Vec<Array1<f64>> = (0..k)
            .into_par_iter()
            .map(|col| {
                let mut bv = Array1::<f64>::zeros(p);
                self.mul_vec_into(factor.column(col), bv.view_mut());
                bv
            })
            .collect();
        for (col, bv) in cols.into_iter().enumerate() {
            out.column_mut(col).assign(&bv);
        }
        out
    }

    /// Compute `trace(F^T B F)` for a `(p x k)` factor matrix `F`.
    fn trace_projected_factor(&self, factor: &Array2<f64>) -> f64 {
        let op_factor = self.mul_mat(factor);
        factor
            .iter()
            .zip(op_factor.iter())
            .map(|(&f, &bf)| f * bf)
            .sum()
    }

    /// Optional stable identity for this operator's action `B`. When `Some`,
    /// the default cached trace / projected-matrix paths memoize the `B · F`
    /// product in the shared [`ProjectedFactorCache`] under a
    /// `(design_id, factor)` key, so repeated projections of the same factor
    /// against the same operator within one outer iteration build `B · F`
    /// once. `None` (the default) disables that reuse: an operator with no
    /// design factor stable across calls cannot key the cache without risking
    /// a stale `B · F`, so it recomputes every time.
    fn projection_design_id(&self) -> Option<usize> {
        None
    }

    fn trace_projected_factor_cached(
        &self,
        factor: &Array2<f64>,
        factor_cache: &ProjectedFactorCache,
    ) -> f64 {
        // The default implementation has no use for the caller-owned cache;
        // verify the cache object carries a positive-size allocation before
        // delegating to the exact path.
        assert!(std::mem::size_of_val(factor_cache) > 0);
        match self.projection_design_id() {
            Some(design_id) => {
                let key = ProjectedFactorKey::from_factor_view(design_id, factor.view());
                let projected = factor_cache.get_or_insert_with(key, || self.mul_mat(factor));
                factor
                    .iter()
                    .zip(projected.iter())
                    .map(|(&f, &bf)| f * bf)
                    .sum()
            }
            None => self.trace_projected_factor(factor),
        }
    }

    /// Compute the exact projected matrix `F^T B F`.
    fn projected_matrix(&self, factor: &Array2<f64>) -> Array2<f64> {
        let op_factor = self.mul_mat(factor);
        gam_linalg::faer_ndarray::fast_atb(factor, &op_factor)
    }

    /// Compute the exact projected matrix `F^T B F`, reusing caller-owned
    /// projection caches when the operator has a shared row/design factor.
    fn projected_matrix_cached(
        &self,
        factor: &Array2<f64>,
        factor_cache: &ProjectedFactorCache,
    ) -> Array2<f64> {
        assert!(std::mem::size_of_val(factor_cache) > 0);
        match self.projection_design_id() {
            Some(design_id) => {
                let key = ProjectedFactorKey::from_factor_view(design_id, factor.view());
                let projected = factor_cache.get_or_insert_with(key, || self.mul_mat(factor));
                gam_linalg::faer_ndarray::fast_atb(factor, projected.as_ref())
            }
            None => self.projected_matrix(factor),
        }
    }

    /// Fill columns `[start, start + out.ncols())` of `B` into `out`.
    fn mul_basis_columns_into(&self, start: usize, mut out: ArrayViewMut2<'_, f64>) {
        let cols = out.ncols();
        let dim = out.nrows();
        assert!(start + cols <= dim);
        let mut basis = Array1::<f64>::zeros(dim);
        for local_col in 0..cols {
            let global_col = start + local_col;
            basis[global_col] = 1.0;
            self.mul_vec_into(basis.view(), out.column_mut(local_col));
            basis[global_col] = 0.0;
        }
    }

    /// Accumulate `scale * B · v` into caller-owned storage.
    fn scaled_add_mul_vec(
        &self,
        v: ArrayView1<'_, f64>,
        scale: f64,
        mut out: ArrayViewMut1<'_, f64>,
    ) {
        if scale == 0.0 {
            return;
        }
        let mut work = Array1::<f64>::zeros(out.len());
        self.mul_vec_into(v, work.view_mut());
        out.scaled_add(scale, &work);
    }

    /// Compute v^T · B · u (bilinear form).
    fn bilinear(&self, v: &Array1<f64>, u: &Array1<f64>) -> f64 {
        let mut bv = Array1::<f64>::zeros(v.len());
        self.mul_vec_into(v.view(), bv.view_mut());
        u.dot(&bv)
    }

    /// Compute v^T · B · u without requiring owned vector inputs.
    fn bilinear_view(&self, v: ArrayView1<'_, f64>, u: ArrayView1<'_, f64>) -> f64 {
        let mut bv = Array1::<f64>::zeros(v.len());
        self.mul_vec_into(v, bv.view_mut());
        u.dot(&bv)
    }

    /// Whether `bilinear_view` is implemented as a direct scalar contraction.
    fn has_fast_bilinear_view(&self) -> bool {
        false
    }

    /// Full dense materialization.
    fn to_dense(&self) -> Array2<f64> {
        let p = self.dim();
        let mut out = Array2::<f64>::zeros((p, p));
        let mut basis = Array1::<f64>::zeros(p);
        for j in 0..p {
            basis[j] = 1.0;
            self.mul_vec_into(basis.view(), out.column_mut(j));
            basis[j] = 0.0;
        }
        out
    }

    /// Whether this operator uses implicit (non-materialized) storage.
    fn is_implicit(&self) -> bool;

    /// If this operator is block-local, returns the block range and local matrix.
    fn block_local_data(&self) -> Option<(&Array2<f64>, usize, usize)> {
        None
    }
}

#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub struct ProjectedFactorKey {
    pub(crate) design_id: usize,
    pub(crate) factor_ptr: usize,
    pub(crate) rows: usize,
    pub(crate) cols: usize,
    pub(crate) row_stride: isize,
    pub(crate) col_stride: isize,
    pub(crate) value_hash: u64,
    pub(crate) value_hash2: u64,
}

impl ProjectedFactorKey {
    pub fn from_factor_view(design_id: usize, factor: ArrayView2<'_, f64>) -> Self {
        let strides = factor.strides();
        let (value_hash, value_hash2) = projected_factor_value_fingerprint(factor);
        Self {
            design_id,
            factor_ptr: factor.as_ptr() as usize,
            rows: factor.nrows(),
            cols: factor.ncols(),
            row_stride: strides[0],
            col_stride: strides[1],
            value_hash,
            value_hash2,
        }
    }

    /// Construct a synthetic, unique-by-`seed` key without going through
    /// [`Self::from_factor_view`]. Used by cache tests that need to inject
    /// fingerprints directly (and deterministically) rather than relying on
    /// ndarray pointer aliasing, which the real constructor keys on.
    pub fn synthetic(seed: u64) -> Self {
        Self {
            design_id: 1,
            factor_ptr: seed as usize,
            rows: 1,
            cols: 1,
            row_stride: 1,
            col_stride: 1,
            value_hash: seed,
            value_hash2: seed.wrapping_mul(31),
        }
    }
}

pub(crate) fn projected_factor_value_fingerprint(factor: ArrayView2<'_, f64>) -> (u64, u64) {
    let mut h1 = 0xcbf2_9ce4_8422_2325_u64;
    let mut h2 = 0x9e37_79b1_85eb_ca87_u64;
    for (idx, value) in factor.iter().enumerate() {
        let bits = value.to_bits();
        let mixed = bits.wrapping_add((idx as u64).wrapping_mul(0x517c_c1b7_2722_0a95));
        h1 ^= mixed;
        h1 = h1.wrapping_mul(0x0000_0100_0000_01b3);
        h2 ^= bits.rotate_left((idx & 63) as u32);
        h2 = h2.wrapping_mul(0x94d0_49bb_1331_11eb).rotate_left(27);
    }
    (h1, h2)
}

/// Memoizer for projected factor products keyed on a `(design, factor)` fingerprint.
pub struct ProjectedFactorCache {
    pub(crate) inner: Mutex<ProjectedFactorCacheInner>,
}

pub(crate) struct ProjectedFactorCacheInner {
    pub(crate) entries: HashMap<ProjectedFactorKey, ProjectedFactorEntry>,
    pub(crate) in_progress: HashMap<ProjectedFactorKey, Arc<ProjectedFactorInProgress>>,
    pub(crate) next_seq: u64,
    pub(crate) total_bytes: usize,
    pub(crate) budget_bytes: usize,
}

pub(crate) struct ProjectedFactorInProgress {
    pub(crate) state: Mutex<Option<ProjectedFactorInProgressState>>,
    pub(crate) ready: Condvar,
    pub(crate) waiter_count: std::sync::atomic::AtomicUsize,
    pub(crate) subscriber_arrived: (Mutex<()>, Condvar),
}

pub(crate) enum ProjectedFactorInProgressState {
    Ready(Arc<Array2<f64>>),
    Failed,
}

pub(crate) struct ProjectedFactorEntry {
    pub(crate) value: Arc<Array2<f64>>,
    pub(crate) bytes: usize,
    pub(crate) last_used: u64,
}

impl Default for ProjectedFactorCache {
    fn default() -> Self {
        Self::with_budget(Self::DEFAULT_BUDGET_BYTES)
    }
}

impl ProjectedFactorCache {
    pub const DEFAULT_BUDGET_BYTES: usize = 2 * 1024 * 1024 * 1024;

    pub fn with_budget(budget_bytes: usize) -> Self {
        Self {
            inner: Mutex::new(ProjectedFactorCacheInner {
                entries: HashMap::new(),
                in_progress: HashMap::new(),
                next_seq: 0,
                total_bytes: 0,
                budget_bytes,
            }),
        }
    }

    pub fn get_or_insert_with(
        &self,
        key: ProjectedFactorKey,
        compute: impl FnOnce() -> Array2<f64>,
    ) -> Arc<Array2<f64>> {
        enum CacheLookup {
            Hit(Arc<Array2<f64>>),
            Wait(Arc<ProjectedFactorInProgress>),
            Compute(Arc<ProjectedFactorInProgress>),
        }

        let lookup = {
            let mut inner = self
                .inner
                .lock()
                .expect("projected factor cache lock poisoned");
            inner.next_seq += 1;
            let now = inner.next_seq;
            if let Some(entry) = inner.entries.get_mut(&key) {
                entry.last_used = now;
                CacheLookup::Hit(entry.value.clone())
            } else if let Some(waiter) = inner.in_progress.get(&key) {
                CacheLookup::Wait(waiter.clone())
            } else {
                let marker = Arc::new(ProjectedFactorInProgress {
                    state: Mutex::new(None),
                    ready: Condvar::new(),
                    waiter_count: std::sync::atomic::AtomicUsize::new(0),
                    subscriber_arrived: (Mutex::new(()), Condvar::new()),
                });
                inner.in_progress.insert(key, marker.clone());
                CacheLookup::Compute(marker)
            }
        };

        match lookup {
            CacheLookup::Hit(value) => value,
            CacheLookup::Wait(marker) => {
                marker
                    .waiter_count
                    .fetch_add(1, std::sync::atomic::Ordering::AcqRel);
                let (lock, cv) = &marker.subscriber_arrived;
                drop(
                    lock.lock()
                        .expect("subscriber-arrived notification lock poisoned"),
                );
                cv.notify_all();
                let mut guard = marker
                    .state
                    .lock()
                    .expect("projected factor in-progress lock poisoned");
                let result = loop {
                    match guard.as_ref() {
                        Some(ProjectedFactorInProgressState::Ready(value)) => {
                            break value.clone();
                        }
                        Some(ProjectedFactorInProgressState::Failed) => {
                            marker
                                .waiter_count
                                .fetch_sub(1, std::sync::atomic::Ordering::AcqRel);
                            reml_contract_panic("projected factor cache producer panicked")
                        }
                        None => {
                            guard = marker
                                .ready
                                .wait(guard)
                                .expect("projected factor in-progress wait poisoned");
                        }
                    }
                };
                marker
                    .waiter_count
                    .fetch_sub(1, std::sync::atomic::Ordering::AcqRel);
                result
            }
            CacheLookup::Compute(marker) => {
                let computed = match catch_unwind(AssertUnwindSafe(|| Arc::new(compute()))) {
                    Ok(value) => value,
                    Err(payload) => {
                        let mut inner = self
                            .inner
                            .lock()
                            .expect("projected factor cache lock poisoned");
                        inner.in_progress.remove(&key);
                        drop(inner);

                        let mut guard = marker
                            .state
                            .lock()
                            .expect("projected factor in-progress lock poisoned");
                        *guard = Some(ProjectedFactorInProgressState::Failed);
                        marker.ready.notify_all();
                        resume_unwind(payload);
                    }
                };
                let bytes = computed.len().saturating_mul(std::mem::size_of::<f64>());
                let mut inner = self
                    .inner
                    .lock()
                    .expect("projected factor cache lock poisoned");
                inner.next_seq += 1;
                let now = inner.next_seq;

                if inner.budget_bytes > 0 && bytes <= inner.budget_bytes {
                    while inner.total_bytes.saturating_add(bytes) > inner.budget_bytes
                        && !inner.entries.is_empty()
                    {
                        let Some(oldest_key) = inner
                            .entries
                            .iter()
                            .min_by_key(|(_, e)| e.last_used)
                            .map(|(k, _)| *k)
                        else {
                            break;
                        };
                        if let Some(removed) = inner.entries.remove(&oldest_key) {
                            inner.total_bytes = inner.total_bytes.saturating_sub(removed.bytes);
                        }
                    }
                }

                let value = if let Some(entry) = inner.entries.get_mut(&key) {
                    entry.last_used = now;
                    entry.value.clone()
                } else {
                    inner.entries.insert(
                        key,
                        ProjectedFactorEntry {
                            value: computed.clone(),
                            bytes,
                            last_used: now,
                        },
                    );
                    inner.total_bytes = inner.total_bytes.saturating_add(bytes);
                    computed
                };
                inner.in_progress.remove(&key);
                drop(inner);

                let mut guard = marker
                    .state
                    .lock()
                    .expect("projected factor in-progress lock poisoned");
                *guard = Some(ProjectedFactorInProgressState::Ready(value.clone()));
                marker.ready.notify_all();
                value
            }
        }
    }

    pub fn len(&self) -> usize {
        self.inner
            .lock()
            .map(|inner| inner.entries.len())
            .unwrap_or(0)
    }

    pub fn total_bytes(&self) -> usize {
        self.inner
            .lock()
            .map(|inner| inner.total_bytes)
            .unwrap_or(0)
    }

    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Test/diagnostic affordance: block until a consumer has subscribed to the
    /// in-progress slot for `key` (i.e. is waiting on the producer), or until
    /// `timeout` elapses. Returns `true` if a subscriber arrived, `false` if the
    /// key has no in-progress slot or the wait timed out.
    ///
    /// This lives on the cache because it reaches into the per-key subscriber
    /// condvar and waiter counter, which are private synchronization internals;
    /// exposing it as a method keeps those fields encapsulated while still
    /// letting downstream tests deterministically order producer/consumer
    /// interleavings.
    pub fn wait_for_subscriber(
        &self,
        key: ProjectedFactorKey,
        timeout: std::time::Duration,
    ) -> bool {
        let marker = {
            let inner = self
                .inner
                .lock()
                .expect("projected factor cache lock poisoned");
            let Some(m) = inner.in_progress.get(&key) else {
                return false;
            };
            Arc::clone(m)
        };
        if marker
            .waiter_count
            .load(std::sync::atomic::Ordering::Acquire)
            > 0
        {
            return true;
        }
        let (lock, cv) = &marker.subscriber_arrived;
        let mut guard = lock
            .lock()
            .expect("subscriber-arrived notification lock poisoned");
        let deadline = std::time::Instant::now() + timeout;
        loop {
            if marker
                .waiter_count
                .load(std::sync::atomic::Ordering::Acquire)
                > 0
            {
                return true;
            }
            let now = std::time::Instant::now();
            if now >= deadline {
                return false;
            }
            let (next_guard, result) = cv
                .wait_timeout(guard, deadline - now)
                .expect("subscriber-arrived wait poisoned");
            guard = next_guard;
            if result.timed_out()
                && marker
                    .waiter_count
                    .load(std::sync::atomic::Ordering::Acquire)
                    == 0
            {
                return false;
            }
        }
    }
}

#[derive(Clone)]
pub struct DenseMatrixHyperOperator {
    pub matrix: Array2<f64>,
}

impl HyperOperator for DenseMatrixHyperOperator {
    fn dim(&self) -> usize {
        self.matrix.nrows()
    }

    fn mul_vec(&self, v: &Array1<f64>) -> Array1<f64> {
        self.matrix.dot(v)
    }

    fn as_any(&self) -> &(dyn Any + 'static) {
        self
    }

    fn mul_vec_view(&self, v: ArrayView1<'_, f64>) -> Array1<f64> {
        self.matrix.dot(&v)
    }

    fn mul_vec_into(&self, v: ArrayView1<'_, f64>, mut out: ArrayViewMut1<'_, f64>) {
        assert_eq!(self.matrix.ncols(), v.len());
        assert_eq!(self.matrix.nrows(), out.len());
        for (row, out_value) in self.matrix.rows().into_iter().zip(out.iter_mut()) {
            *out_value = row.dot(&v);
        }
    }

    fn mul_basis_columns_into(&self, start: usize, mut out: ArrayViewMut2<'_, f64>) {
        let end = start + out.ncols();
        assert!(end <= self.matrix.ncols());
        out.assign(&self.matrix.slice(ndarray::s![.., start..end]));
    }

    fn scaled_add_mul_vec(
        &self,
        v: ArrayView1<'_, f64>,
        scale: f64,
        mut out: ArrayViewMut1<'_, f64>,
    ) {
        assert_eq!(self.matrix.ncols(), v.len());
        assert_eq!(self.matrix.nrows(), out.len());
        if scale == 0.0 {
            return;
        }
        for (row, out_value) in self.matrix.rows().into_iter().zip(out.iter_mut()) {
            *out_value += scale * row.dot(&v);
        }
    }

    fn bilinear(&self, v: &Array1<f64>, u: &Array1<f64>) -> f64 {
        dense::bilinear(&self.matrix, v.view(), u.view())
    }

    fn bilinear_view(&self, v: ArrayView1<'_, f64>, u: ArrayView1<'_, f64>) -> f64 {
        dense::bilinear(&self.matrix, v, u)
    }

    fn to_dense(&self) -> Array2<f64> {
        self.matrix.clone()
    }

    fn is_implicit(&self) -> bool {
        false
    }
}

#[derive(Clone)]
pub struct BlockLocalDrift {
    pub local: Array2<f64>,
    pub start: usize,
    pub end: usize,
    pub total_dim: usize,
}

impl HyperOperator for BlockLocalDrift {
    fn dim(&self) -> usize {
        self.total_dim
    }

    fn mul_vec(&self, v: &Array1<f64>) -> Array1<f64> {
        assert_eq!(v.len(), self.total_dim);
        let mut out = Array1::zeros(self.total_dim);
        self.mul_vec_into(v.view(), out.view_mut());
        out
    }

    fn as_any(&self) -> &(dyn Any + 'static) {
        self
    }

    fn mul_vec_into(&self, v: ArrayView1<'_, f64>, mut out: ArrayViewMut1<'_, f64>) {
        assert_eq!(v.len(), self.total_dim);
        assert_eq!(out.len(), self.total_dim);
        out.fill(0.0);
        let v_block = v.slice(ndarray::s![self.start..self.end]);
        let mut out_block = out.slice_mut(ndarray::s![self.start..self.end]);
        dense::matvec_into(&self.local, v_block, out_block.view_mut());
    }

    fn scaled_add_mul_vec(
        &self,
        v: ArrayView1<'_, f64>,
        scale: f64,
        mut out: ArrayViewMut1<'_, f64>,
    ) {
        assert_eq!(v.len(), self.total_dim);
        assert_eq!(out.len(), self.total_dim);
        if scale == 0.0 {
            return;
        }
        let v_block = v.slice(ndarray::s![self.start..self.end]);
        let out_block = out.slice_mut(ndarray::s![self.start..self.end]);
        dense::matvec_scaled_add_into(&self.local, v_block, scale, out_block);
    }

    fn bilinear(&self, v: &Array1<f64>, u: &Array1<f64>) -> f64 {
        self.bilinear_view(v.view(), u.view())
    }

    fn bilinear_view(&self, v: ArrayView1<'_, f64>, u: ArrayView1<'_, f64>) -> f64 {
        assert_eq!(v.len(), self.total_dim);
        assert_eq!(u.len(), self.total_dim);
        let v_block = v.slice(ndarray::s![self.start..self.end]);
        let u_block = u.slice(ndarray::s![self.start..self.end]);
        dense::bilinear(&self.local, v_block, u_block)
    }

    fn to_dense(&self) -> Array2<f64> {
        let p = self.total_dim;
        let mut out = Array2::zeros((p, p));
        out.slice_mut(ndarray::s![self.start..self.end, self.start..self.end])
            .assign(&self.local);
        out
    }

    fn is_implicit(&self) -> bool {
        false
    }

    fn block_local_data(&self) -> Option<(&Array2<f64>, usize, usize)> {
        Some((&self.local, self.start, self.end))
    }
}

#[derive(Clone)]
pub struct HyperCoordDrift {
    pub dense: Option<Array2<f64>>,
    pub block_local: Option<BlockLocalDrift>,
    pub operator: Option<Arc<dyn HyperOperator>>,
}

impl HyperCoordDrift {
    pub fn none() -> Self {
        Self {
            dense: None,
            block_local: None,
            operator: None,
        }
    }

    pub fn from_dense(dense: Array2<f64>) -> Self {
        Self {
            dense: Some(dense),
            block_local: None,
            operator: None,
        }
    }

    pub fn from_operator(operator: Arc<dyn HyperOperator>) -> Self {
        Self {
            dense: None,
            block_local: None,
            operator: Some(operator),
        }
    }

    pub fn from_parts(
        dense: Option<Array2<f64>>,
        operator: Option<Arc<dyn HyperOperator>>,
    ) -> Self {
        let dense = dense.filter(|mat| !(operator.is_some() && mat.is_empty()));
        Self {
            dense,
            block_local: None,
            operator,
        }
    }

    pub fn from_block_local_and_operator(
        local: Array2<f64>,
        start: usize,
        end: usize,
        total_dim: usize,
        operator: Option<Arc<dyn HyperOperator>>,
    ) -> Self {
        Self {
            dense: None,
            block_local: Some(BlockLocalDrift {
                local,
                start,
                end,
                total_dim,
            }),
            operator,
        }
    }

    pub fn has_operator(&self) -> bool {
        self.operator.is_some()
    }

    pub fn uses_operator_fast_path(&self) -> bool {
        self.operator.is_some() || self.block_local.is_some()
    }

    pub fn operator_ref(&self) -> Option<&dyn HyperOperator> {
        self.operator.as_ref().map(Arc::as_ref)
    }

    pub fn materialize(&self) -> Array2<f64> {
        let p = self.infer_dim();
        if p == 0 {
            return Array2::zeros((0, 0));
        }
        let mut out = self.dense.clone().unwrap_or_else(|| Array2::zeros((p, p)));
        if let Some(bl) = &self.block_local {
            out.slice_mut(ndarray::s![bl.start..bl.end, bl.start..bl.end])
                .scaled_add(1.0, &bl.local);
        }
        if let Some(op) = &self.operator {
            out += &op.to_dense();
        }
        out
    }

    pub fn apply(&self, v: &Array1<f64>) -> Array1<f64> {
        let mut out = Array1::zeros(v.len());
        self.scaled_add_apply(v.view(), 1.0, &mut out);
        out
    }

    pub fn scaled_add_apply(&self, v: ArrayView1<'_, f64>, scale: f64, out: &mut Array1<f64>) {
        assert_eq!(v.len(), out.len());
        if scale == 0.0 {
            return;
        }
        if let Some(dense) = &self.dense {
            dense::matvec_scaled_add_into(dense, v, scale, out.view_mut());
        }
        if let Some(bl) = &self.block_local {
            let v_block = v.slice(ndarray::s![bl.start..bl.end]);
            let out_block = out.slice_mut(ndarray::s![bl.start..bl.end]);
            dense::matvec_scaled_add_into(&bl.local, v_block, scale, out_block);
        }
        if let Some(op) = &self.operator {
            op.scaled_add_mul_vec(v, scale, out.view_mut());
        }
    }

    pub(crate) fn infer_dim(&self) -> usize {
        if let Some(d) = &self.dense {
            return d.nrows();
        }
        if let Some(op) = &self.operator {
            return op.dim();
        }
        if let Some(bl) = &self.block_local {
            return bl.total_dim;
        }
        0
    }
}

#[derive(Clone)]
pub struct HyperCoord {
    pub a: f64,
    pub g: Array1<f64>,
    pub drift: HyperCoordDrift,
    pub ld_s: f64,
    pub b_depends_on_beta: bool,
    pub is_penalty_like: bool,
    pub firth_g: Option<Array1<f64>>,
    pub tk_eta_fixed: Option<Array1<f64>>,
    pub tk_x_fixed: Option<Array2<f64>>,
}

#[derive(Clone)]
pub struct HyperCoordPair {
    pub a: f64,
    pub g: Array1<f64>,
    pub b_mat: Array2<f64>,
    pub b_operator: Option<Arc<dyn HyperOperator>>,
    pub ld_s: f64,
}

/// Fallible result of computing one second-order fixed-β coordinate pair.
///
/// Pair assembly may call an immutable family workspace whose shape and
/// numerical validation are deliberately error-capable. Keeping that failure
/// in the callback contract lets dense and operator Hessian consumers stop
/// with the original evidence instead of converting it into a panic.
pub type HyperCoordPairResult = Result<HyperCoordPair, String>;

/// Shared-ownership callback computing a second-order fixed-β
/// [`HyperCoordPairResult`] for a coordinate pair `(i, j)`.
///
/// `Arc` (not `Box`) so the same callback can be cloned into a derived
/// `InnerSolution` — notably the tangent-projected solution built under active
/// inequality constraints, which must carry the very same pair callbacks
/// through to `ValueGradientHessian` outer-Hessian assembly. The pair objects
/// are p-space; every consumer contracts them through the (possibly
/// tangent-wrapped) Hessian operator, which applies the `ZᵀMZ` / `Z H_T⁻¹ Zᵀ`
/// projection internally, so a clone-through is mathematically exact.
pub type HyperCoordPairFn = Arc<dyn Fn(usize, usize) -> HyperCoordPairResult + Send + Sync>;

impl HyperCoordPair {
    pub fn zero() -> Self {
        Self {
            a: 0.0,
            g: Array1::zeros(0),
            b_mat: Array2::zeros((0, 0)),
            b_operator: None,
            ld_s: 0.0,
        }
    }
}

#[derive(Clone)]
pub enum DriftDerivResult {
    Dense(Array2<f64>),
    Operator(Arc<dyn HyperOperator>),
}

impl std::fmt::Debug for DriftDerivResult {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Dense(matrix) => f
                .debug_tuple("Dense")
                .field(&format_args!("{}x{}", matrix.nrows(), matrix.ncols()))
                .finish(),
            Self::Operator(_) => f
                .debug_tuple("Operator")
                .field(&"<hyper-operator>")
                .finish(),
        }
    }
}

impl DriftDerivResult {
    pub fn into_operator(self) -> Arc<dyn HyperOperator> {
        match self {
            Self::Dense(matrix) => Arc::new(DenseMatrixHyperOperator { matrix }),
            Self::Operator(operator) => operator,
        }
    }

    pub fn apply(&self, v: &Array1<f64>) -> Array1<f64> {
        match self {
            Self::Dense(matrix) => matrix.dot(v),
            Self::Operator(operator) => operator.mul_vec(v),
        }
    }
}

pub type FixedDriftDerivFn =
    Box<dyn Fn(usize, &Array1<f64>) -> Result<Option<DriftDerivResult>, String> + Send + Sync>;

/// Shared-ownership form of [`FixedDriftDerivFn`] used for `InnerSolution`
/// storage, so the same `M_i[u] = D_β B_i[u]` callback can be cloned into a
/// derived (tangent-projected) solution. Construction sites still hand back a
/// `Box` ([`FixedDriftDerivFn`]); storage re-tags it via `Arc::from` (free).
/// The drift `M` is a p-space matrix that every consumer contracts through the
/// (tangent-wrapped) Hessian operator's `trace_logdet_*`, so the clone-through
/// is exact under projection.
pub type SharedFixedDriftDerivFn =
    Arc<dyn Fn(usize, &Array1<f64>) -> Result<Option<DriftDerivResult>, String> + Send + Sync>;

pub struct ContractedPsiSecondOrder {
    pub objective: Array1<f64>,
    pub score: Array2<f64>,
    pub hessian: Vec<DriftDerivResult>,
    pub ld_s: Array1<f64>,
}

pub type ContractedPsiSecondOrderFn =
    Arc<dyn Fn(&[f64]) -> Result<Option<ContractedPsiSecondOrder>, String> + Send + Sync>;
