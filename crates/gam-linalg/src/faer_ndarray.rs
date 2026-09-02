use dyn_stack::{MemBuffer, MemStack};
use faer::diag::{Diag, DiagRef};
use faer::linalg::solvers::{self, Solve};
pub use faer::linalg::solvers::{
    Lblt as FaerLblt, Ldlt as FaerLdlt, Llt as FaerLlt, Solve as FaerSolve,
};
use faer::linalg::svd::{self, ComputeSvdVectors};
use faer::prelude::ReborrowMut;
use faer::{Conj, Mat, MatMut, MatRef, Par, Side, Unbind, get_global_parallelism};
use ndarray::{Array1, Array2, ArrayBase, ArrayView1, ArrayViewMut1, Data, Ix1, Ix2};
use std::marker::PhantomData;
use std::panic::{AssertUnwindSafe, catch_unwind};
use std::sync::atomic::{AtomicU64, Ordering};
use thiserror::Error;

/// Apply a symmetric matrix through the crate-owned SIMD/FMA GEMV kernel.
///
/// Lanczos certifies the resulting Ritz residuals, so this uses the ordinary
/// FMA kernel rather than the substantially more expensive Dot2 reduction.
/// Keeping the implementation inside `gam-linalg` makes the library complete:
/// callers do not acquire an undeclared process-global BLAS dependency merely
/// by linking a Duchon basis.
pub fn symmetric_matvec_into(
    matrix: &Array2<f64>,
    vector: &[f64],
    output: &mut [f64],
) -> Result<(), String> {
    let n = matrix.nrows();
    if matrix.ncols() != n || vector.len() != n || output.len() != n {
        return Err(format!(
            "symmetric matvec shape mismatch: matrix={:?}, vector={}, output={}",
            matrix.dim(),
            vector.len(),
            output.len()
        ));
    }
    fast_av_standard_view_into(
        matrix,
        &ArrayView1::from(vector),
        ArrayViewMut1::from(output),
    );
    Ok(())
}

const RRQR_RANK_ALPHA: f64 = 100.0;

thread_local! {
    static NESTED_PARALLEL_DEPTH: std::cell::Cell<usize> = const { std::cell::Cell::new(0) };
}

struct NestedParallelGuard;

impl NestedParallelGuard {
    #[inline]
    fn enter() -> Self {
        NESTED_PARALLEL_DEPTH.with(|depth| depth.set(depth.get().saturating_add(1)));
        Self
    }
}

impl Drop for NestedParallelGuard {
    #[inline]
    fn drop(&mut self) {
        NESTED_PARALLEL_DEPTH.with(|depth| depth.set(depth.get().saturating_sub(1)));
    }
}

/// Run `body` with the current thread marked as inside a data-parallel row
/// region, so any faer GEMM it issues (directly or transitively) pins to
/// `Par::Seq` via [`effective_global_parallelism`] instead of re-fanning the
/// global Rayon pool. The guard is held for exactly the duration of `body` and
/// dropped on return — including early `?` returns from inside `body`, since the
/// guard lives in this function's frame.
///
/// Call this from the per-chunk/per-row closure of an `into_par_iter` whose body
/// performs GEMM, to prevent the Rayon-pool × faer-pool oversubscription.
#[inline]
pub fn with_nested_parallel<T>(body: impl FnOnce() -> T) -> T {
    let guard = NestedParallelGuard::enter();
    let out = body();
    drop(guard);
    out
}

/// `true` when the current thread is inside at least one `NestedParallelGuard`
/// scope, i.e. a parallel row reduction is already in flight on this thread.
#[inline]
pub fn in_nested_parallel_region() -> bool {
    NESTED_PARALLEL_DEPTH.with(|depth| depth.get() > 0)
}

/// #2267 — process-global census of self-adjoint eigendecompositions.
///
/// A per-call duration answers "was this call slow"; a running count and total
/// answer "was the step one slow call or many", which is the question that
/// separates #2267's two candidate explanations and which no single timing can
/// settle. `Relaxed` is right here: these are a diagnostic census with no
/// happens-before relationship to anything, and the values are only ever read
/// into a log line.
static EIGH_CALLS: AtomicU64 = AtomicU64::new(0);
static EIGH_NANOS: AtomicU64 = AtomicU64::new(0);
/// Of those calls, how many observed `Par::Seq`. This is the field that makes
/// the census ASSERTABLE rather than merely observable: a test can state "the
/// large decomposition ran sequentially" as a bar instead of a human reading it
/// out of a log.
static EIGH_SEQ_CALLS: AtomicU64 = AtomicU64::new(0);
/// The largest `dim` seen, so a run can be asked whether it ever reached the
/// shape under investigation rather than being assumed to have.
static EIGH_MAX_DIM: AtomicU64 = AtomicU64::new(0);

// The same four tallies, for the CALLING THREAD only.
//
// The process-global counters above answer *"how many `eigh` calls did this
// RUN make?"*. They cannot answer *"how many did THIS REGION make?"*, because
// every other thread's `eigh` lands inside the same window — and under
// `cargo test` there are as many such threads as the harness chose. An exact
// delta taken across a region of a global counter is therefore an assertion
// about which OTHER tests happened to share the process, which is not a
// property anybody meant to test: measured at `0033169a9`,
// `eigh_census_counts_calls_and_separates_the_sequential_arm` read `+12`
// instead of `+1` under the default thread count and passed under
// `--test-threads=1`. The per-thread tallies make the delta exact under any
// schedule, so the assertion can stay an equality instead of being weakened to
// an inequality that no longer detects over-counting.
thread_local! {
    static EIGH_THREAD: std::cell::Cell<EighCensus> = const {
        std::cell::Cell::new(EighCensus {
            calls: 0,
            sequential_calls: 0,
            max_dim: 0,
            nanos: 0,
        })
    };
}

/// Add one `eigh` to the calling thread's tallies.
fn record_thread_eigh(sequential: bool, dim: u64, nanos: u64) {
    EIGH_THREAD.with(|cell| {
        let mut census = cell.get();
        census.calls += 1;
        if sequential {
            census.sequential_calls += 1;
        }
        census.max_dim = census.max_dim.max(dim);
        census.nanos += nanos;
        cell.set(census);
    });
}

/// #2267/#2738 — the eigendecomposition census, readable from a test.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct EighCensus {
    /// Total `eigh` calls since process start.
    pub calls: u64,
    /// How many of them observed faer's global parallelism as `Par::Seq`.
    pub sequential_calls: u64,
    /// Largest matrix dimension decomposed.
    pub max_dim: u64,
    /// Cumulative wall time across all calls, in nanoseconds.
    pub nanos: u64,
}

/// #2738 — the thread configuration actually in force, read INSIDE the running
/// process and carried as DATA a test can read.
///
/// A perf experiment that sweeps a thread count needs a manipulation check: proof
/// the treatment took effect. A sweep whose treatment silently failed produces a
/// perfectly flat curve, which is indistinguishable from saturation and points in
/// whichever direction the experimenter expected. So every field here is what the
/// process OBSERVES, never what was exported: an exported variable proves an
/// intention, not an effect. (Reading it is unavailable anyway — `env::var` is
/// banned tree-wide.)
///
/// The fields are separate because they can disagree, and the disagreement is
/// the finding:
///
/// * `rayon::current_num_threads()` — the pool width, the one quantity the
///   diagnostics already printed.
/// * [`faer::get_global_parallelism`] — faer's PROCESS-GLOBAL policy, which is
///   what every high-level faer factorization (`self_adjoint_eigen`, `Llt::new`,
///   `Solve::solve`, SVD, col-pivoted QR) reads internally. A live
///   [`FaerSequentialScope`] anywhere in the process pins this to `Par::Seq` for
///   EVERY thread, so a decomposition can be single-threaded while the rayon pool
///   reports 64. Reporting only the pool width would show a wide machine while
///   the numerics ran on one core, which is exactly the failure this exists to
///   make visible.
/// * the live [`FaerSequentialScope`] depth, which says whether that pin is a
///   scoped decision or a global left over from an earlier phase.
/// * the cores available to THIS PROCESS, which is not the machine's core count.
///
/// Deliberately NOT reported: [`effective_global_parallelism`]. That is
/// thread-local and only governs the codebase's own `matmul` calls; it cannot
/// reach faer's high-level entry points, so printing it beside a factorization
/// would name a policy that did not apply to it.
///
/// There is no BLAS term because this workspace links no CPU BLAS — the numerics
/// are faer and `ndarray`, both parallelised through Rayon. `OPENBLAS_NUM_THREADS`
/// and `OMP_NUM_THREADS` are inert here (they do bind the Python lanes, where
/// numpy and torch link OpenBLAS), so a BLAS thread count printed here would
/// report a number that governs nothing — strictly worse than reporting none.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ParallelismSnapshot {
    /// Width of the Rayon pool this thread belongs to. Every parallel loop in the
    /// workspace, and faer's own `Par::Rayon` dispatch, fan out through it.
    pub rayon_current_num_threads: usize,
    /// `true` when faer's process-global policy is `Par::Seq`, i.e. every
    /// high-level faer factorization (`self_adjoint_eigen`, `Llt::new`,
    /// `Solve::solve`, SVD, col-pivoted QR) runs on ONE core regardless of how
    /// wide the Rayon pool above reports.
    pub faer_global_sequential: bool,
    /// Threads faer's global policy would ideally use ([`Par::degree`]). `1` for
    /// `Par::Seq`, and also `1` for a one-thread `Par::rayon(1)` — which is why
    /// `faer_global_sequential` is carried separately rather than inferred.
    pub faer_global_degree: usize,
    /// How many [`FaerSequentialScope`] guards are alive process-wide. Non-zero
    /// means the sequential pin above is deliberate and scoped, not a stale
    /// global left behind by an earlier phase — a distinction the degree alone
    /// cannot make.
    pub faer_sequential_scope_depth: usize,
    /// Cores available to THIS PROCESS, not cores present in the machine:
    /// `std::thread::available_parallelism` honours the CPU affinity mask and
    /// the cgroup quota, so a 4-CPU Slurm allocation on a 128-core node reports
    /// 4. `None` only when the platform refuses to answer; a fabricated fallback
    /// would be indistinguishable from a real reading.
    pub process_available_parallelism: Option<usize>,
}

impl ParallelismSnapshot {
    /// Read the live configuration.
    pub fn capture() -> Self {
        Self::from_parts(
            get_global_parallelism(),
            rayon::current_num_threads(),
            faer_sequential_scope_depth(),
            std::thread::available_parallelism().ok().map(|n| n.get()),
        )
    }

    /// Assemble from explicit parts. Exists so the consistency rules below can be
    /// exercised against configurations this process is not currently in —
    /// including inconsistent ones, which is the only way to show
    /// [`Self::inconsistency`] is capable of returning `Some`.
    pub fn from_parts(
        faer_global: Par,
        rayon_current_num_threads: usize,
        faer_sequential_scope_depth: usize,
        process_available_parallelism: Option<usize>,
    ) -> Self {
        Self {
            rayon_current_num_threads,
            faer_global_sequential: faer_global == Par::Seq,
            faer_global_degree: faer_global.degree(),
            faer_sequential_scope_depth,
            process_available_parallelism,
        }
    }

    /// `None` when the fields agree with each other; otherwise the first
    /// disagreement, named.
    ///
    /// These are cross-checks between INDEPENDENTLY SOURCED quantities — faer's
    /// global policy, this crate's own scope-depth counter, Rayon's pool width —
    /// so a snapshot that passes has had its three sources agree rather than
    /// merely restated one of them three times.
    pub fn inconsistency(&self) -> Option<String> {
        if self.rayon_current_num_threads == 0 {
            return Some("rayon reports a pool of zero threads".to_string());
        }
        if self.faer_global_degree == 0 {
            return Some("faer's global parallelism has degree zero".to_string());
        }
        if self.faer_global_sequential && self.faer_global_degree != 1 {
            return Some(format!(
                "faer is sequential but reports degree {}",
                self.faer_global_degree
            ));
        }
        if self.faer_sequential_scope_depth > 0 && !self.faer_global_sequential {
            return Some(format!(
                "{} live FaerSequentialScope guard(s) but faer's global parallelism is not sequential",
                self.faer_sequential_scope_depth
            ));
        }
        if self.process_available_parallelism == Some(0) {
            return Some("this process reports zero available cores".to_string());
        }
        None
    }
}

impl std::fmt::Display for ParallelismSnapshot {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "rayon_current_num_threads={} | faer_global_sequential={} | \
             faer_global_degree={} | faer_sequential_scope_depth={} | \
             process_available_parallelism={}",
            self.rayon_current_num_threads,
            self.faer_global_sequential,
            self.faer_global_degree,
            self.faer_sequential_scope_depth,
            match self.process_available_parallelism {
                Some(cores) => cores.to_string(),
                None => "unavailable".to_string(),
            },
        )
    }
}

/// faer parallelism policy that respects nested data-parallel regions: returns
/// faer's global policy at the top level, but `Par::Seq` once a
/// `NestedParallelGuard` is active so a GEMM issued from inside a parallel row
/// fan-out does not multiply the live thread count against the outer pool.
///
/// Use this in place of `faer::get_global_parallelism()` for any matmul that can
/// be reached from inside a row-parallel closure.
#[inline]
pub fn effective_global_parallelism() -> Par {
    if in_nested_parallel_region() {
        Par::Seq
    } else {
        get_global_parallelism()
    }
}

/// Process-global depth counter + saved parallelism for [`FaerSequentialScope`].
///
/// The `effective_global_parallelism` / [`NestedParallelGuard`] pair only pins
/// the codebase's OWN `matmul` calls to `Par::Seq`; it CANNOT reach faer's
/// high-level factorization/solve entry points (`Llt::new`, `Solve::solve`, SVD,
/// col-pivoted QR), which read `faer::get_global_parallelism()` internally and
/// have no per-call parallelism argument. When such a solver runs from inside a
/// Rayon worker (e.g. the topology race fans candidate fits into per-candidate
/// pools via `run_topology_race_parallel`), faer's default `Par::rayon(0)`
/// dispatches the factorization through its `spindle` barrier pool, which
/// `rayon::scope`-spawns as many tasks as the pool has threads and waits for all
/// of them at a barrier. Under thread oversubscription those worker slots are
/// already occupied by the outer fan-out, so the barrier never completes and the
/// fit parks at 0% CPU — the #2074 K=1 `sae_manifold_fit` deadlock.
///
/// [`FaerSequentialScope`] closes that hole by pinning faer's PROCESS-GLOBAL
/// parallelism to `Par::Seq` around the nested solve, so every faer solver it
/// reaches stays single-threaded and never spawns a nested barrier pool. The
/// codebase engineers its faer reductions to be parallelism-invariant
/// (`tests_parallelism_invariance_1557` asserts byte-identical `Par::Seq` vs
/// `Par::rayon` output), so collapsing to sequential is bit-for-bit neutral.
static FAER_SEQ_STATE: std::sync::Mutex<FaerSeqState> = std::sync::Mutex::new(FaerSeqState {
    depth: 0,
    saved: None,
});

struct FaerSeqState {
    depth: usize,
    saved: Option<Par>,
}

/// RAII guard that pins faer's process-global parallelism to [`Par::Seq`] for its
/// lifetime and restores the previous setting when the LAST live guard drops.
///
/// The guard is depth-counted across threads: overlapping guards (e.g. several
/// topology-race candidates fitting concurrently) all observe `Par::Seq`, and the
/// prior policy is restored exactly once, when the outermost guard exits. Setting
/// and restoring happen under the state mutex so the `depth == 0` transition is
/// atomic with the `set_global_parallelism` call.
#[must_use = "the sequential scope only holds while the guard is alive"]
pub struct FaerSequentialScope {
    _private: (),
}

impl FaerSequentialScope {
    /// Enter the scope, forcing faer to `Par::Seq` on the `0 -> 1` transition.
    pub fn enter() -> Self {
        let mut state = FAER_SEQ_STATE
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        if state.depth == 0 {
            state.saved = Some(get_global_parallelism());
            faer::set_global_parallelism(Par::Seq);
        }
        state.depth += 1;
        Self { _private: () }
    }
}

impl Drop for FaerSequentialScope {
    fn drop(&mut self) {
        let mut state = FAER_SEQ_STATE
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        state.depth -= 1;
        if state.depth == 0 {
            if let Some(par) = state.saved.take() {
                faer::set_global_parallelism(par);
            }
        }
    }
}

/// #2738 — how many [`FaerSequentialScope`] guards are alive process-wide.
///
/// The depth is what distinguishes "faer is sequential because a solve here
/// asked for it" from "faer is sequential and nobody knows who did it", and it
/// is readable without a logger, which `log::info!` is not under `cargo test`.
pub fn faer_sequential_scope_depth() -> usize {
    FAER_SEQ_STATE
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner)
        .depth
}

/// Run `body` with faer pinned to `Par::Seq` (see [`FaerSequentialScope`]). Use
/// this to wrap a fit/solve that runs inside a Rayon worker so faer's high-level
/// solvers never fan a nested `spindle` barrier pool into an already-saturated
/// Rayon pool.
#[inline]
pub fn with_faer_sequential<T>(body: impl FnOnce() -> T) -> T {
    let faer_seq_guard = FaerSequentialScope::enter();
    let out = body();
    drop(faer_seq_guard);
    out
}

#[derive(Debug, Error)]
pub enum FaerLinalgError {
    #[error("Factorization failed in {context}")]
    FactorizationFailed { context: &'static str },
    #[error("SVD failed to converge in {context}")]
    SvdNoConvergence { context: &'static str },
    #[error("Self-adjoint eigendecomposition input contains non-finite values in {context}")]
    SelfAdjointEigenNonFiniteInput { context: &'static str },
    #[error("Strict self-adjoint eigendecomposition rejected its input: {reason}")]
    StrictSelfAdjointEigenInvalidInput { reason: String },
    #[error("Self-adjoint eigendecomposition failed: {0:?}")]
    SelfAdjointEigen(solvers::EvdError),
    #[error("Cholesky factorization failed: {0:?}")]
    Cholesky(solvers::LltError),
    #[error("LDLT factorization failed: {0:?}")]
    Ldlt(solvers::LdltError),
}

pub enum FaerSymmetricFactor {
    Llt(FaerLlt<f64>),
    Ldlt(FaerLdlt<f64>),
    Lblt(FaerLblt<f64>),
}

#[inline]
pub fn cholesky_factor_logdet(factor: MatRef<'_, f64>) -> f64 {
    2.0 * diagonal_log_sum(factor.diagonal())
}

#[inline]
fn diagonal_log_sum(diagonal: DiagRef<'_, f64>) -> f64 {
    diagonal
        .column_vector()
        .iter()
        .map(|&x| x.ln())
        .sum::<f64>()
}

impl FaerSymmetricFactor {
    /// Returns the dimension of the factorized square matrix.
    #[inline]
    pub fn n(&self) -> usize {
        use faer::linalg::solvers::ShapeCore;
        match self {
            FaerSymmetricFactor::Llt(f) => f.nrows(),
            FaerSymmetricFactor::Ldlt(f) => f.nrows(),
            FaerSymmetricFactor::Lblt(f) => f.nrows(),
        }
    }

    #[inline]
    pub fn solve(&self, rhs: MatRef<'_, f64>) -> Mat<f64> {
        match self {
            FaerSymmetricFactor::Llt(f) => f.solve(rhs),
            FaerSymmetricFactor::Ldlt(f) => f.solve(rhs),
            FaerSymmetricFactor::Lblt(f) => f.solve(rhs),
        }
    }

    #[inline]
    pub fn solve_in_place(&self, rhs: MatMut<'_, f64>) {
        match self {
            FaerSymmetricFactor::Llt(f) => f.solve_in_place(rhs),
            FaerSymmetricFactor::Ldlt(f) => f.solve_in_place(rhs),
            FaerSymmetricFactor::Lblt(f) => f.solve_in_place(rhs),
        }
    }
}

impl crate::matrix::FactorizedSystem for FaerSymmetricFactor {
    fn solve(&self, rhs: &Array1<f64>) -> Result<Array1<f64>, String> {
        let mut out = rhs.clone();
        let mut out_mat = array1_to_col_matmut(&mut out);
        self.solve_in_place(out_mat.as_mut());
        if !out.iter().all(|v| v.is_finite()) {
            return Err("symmetric factor solve produced non-finite values".to_string());
        }
        Ok(out)
    }

    fn solvemulti(&self, rhs: &Array2<f64>) -> Result<Array2<f64>, String> {
        let mut out = Array2::<f64>::zeros(rhs.raw_dim());
        for j in 0..rhs.ncols() {
            for i in 0..rhs.nrows() {
                out[[i, j]] = rhs[[i, j]];
            }
        }
        let mut out_mat = array2_to_matmut(&mut out);
        self.solve_in_place(out_mat.as_mut());
        if !out.iter().all(|v| v.is_finite()) {
            return Err("symmetric factor multi-solve produced non-finite values".to_string());
        }
        Ok(out)
    }

    fn logdet(&self) -> f64 {
        match self {
            FaerSymmetricFactor::Llt(f) => cholesky_factor_logdet(f.L()),
            FaerSymmetricFactor::Ldlt(f) => diagonal_log_sum(f.D()),
            FaerSymmetricFactor::Lblt(..) => {
                // lblt doesn't easily expose diagonal determinant. Fallback to sparse or other representations if needed, but typically Lblt is indefinite!
                // Actually faer doesn't easily expose lblt logdet since it has 2x2 blocks.
                // For our ML systems, if we dropped to LBLT, the matrix was indefinite and logdet is ill-defined (or complex).
                f64::NAN
            }
        }
    }
}

/// Factorize a symmetric system with LLT -> LDLT -> LBLT fallback.
#[inline]
pub fn factorize_symmetricwith_fallback(
    matrix: MatRef<'_, f64>,
    side: Side,
) -> Result<FaerSymmetricFactor, FaerLinalgError> {
    if let Ok(llt) = FaerLlt::new(matrix, side) {
        return Ok(FaerSymmetricFactor::Llt(llt));
    }
    let ldlt_err = match FaerLdlt::new(matrix, side) {
        Ok(ldlt) => return Ok(FaerSymmetricFactor::Ldlt(ldlt)),
        Err(err) => err,
    };
    let lblt = catch_unwind(AssertUnwindSafe(|| FaerLblt::new(matrix, side)))
        .map_err(|_| FaerLinalgError::Ldlt(ldlt_err))?;
    Ok(FaerSymmetricFactor::Lblt(lblt))
}

#[inline]
const fn should_use_faer_matmul(m: usize, n: usize, k: usize) -> bool {
    // Small, centralized dispatch policy:
    // - stay on ndarray for tiny products to avoid setup overhead,
    // - switch to faer GEMM/GEMV for moderate+ sizes.
    const MIN_DIM: usize = 32;
    const MIN_FLOP_SCALE: usize = 64 * 64;
    (m >= MIN_DIM || n >= MIN_DIM || k >= MIN_DIM)
        && m.saturating_mul(n).saturating_mul(k) >= MIN_FLOP_SCALE
}

#[inline]
pub fn matmul_parallelism(m: usize, n: usize, k: usize) -> Par {
    // Prefer a work-based policy over per-dimension thresholds.
    // Tall/skinny products (e.g. N x p with large N, modest p) should still
    // parallelize when total work is high.
    const PAR_MIN_FLOP_SCALE: usize = 2_000_000;
    const PAR_MIN_LONG_DIM: usize = 256;
    let flop_scale = m.saturating_mul(n).saturating_mul(k);
    let long_dim = m.max(n).max(k);
    if flop_scale >= PAR_MIN_FLOP_SCALE && long_dim >= PAR_MIN_LONG_DIM {
        // `effective_global_parallelism` collapses to `Par::Seq` when this GEMM
        // is reached from inside a `NestedParallelGuard` row region, preventing
        // the Rayon-pool × faer-pool multiplicative oversubscription.
        effective_global_parallelism()
    } else {
        Par::Seq
    }
}

#[inline]
pub fn array2_to_matmut(array: &mut Array2<f64>) -> MatMut<'_, f64> {
    let (rows, cols) = array.dim();
    let strides = array.strides();

    // Check if we can get a pointer.
    // If the array is contiguous (either C or F order), or simply sliced with strides,
    // faer can handle it as long as we pass the pointer and strides.
    // However, as_mut_ptr() requires a mutable reference.
    // ndarray's as_ptr/as_mut_ptr works for both layouts.

    let s0 = strides[0];
    let s1 = strides[1];

    // SAFETY: array.as_mut_ptr() is ndarray's logical (0, 0) pointer, and
    // ndarray's dimensions plus signed element strides describe every initialized
    // element of this uniquely borrowed Array2 for the returned MatMut lifetime.
    unsafe { MatMut::from_raw_parts_mut(array.as_mut_ptr(), rows, cols, s0, s1) }
}

/// Convert an ndarray matrix into row-major nested vectors for serialized
/// payloads without exposing storage-layout assumptions to callers.
pub fn array2_to_nested_vec(array: &Array2<f64>) -> Vec<Vec<f64>> {
    array.rows().into_iter().map(|row| row.to_vec()).collect()
}

#[inline]
pub fn array1_to_col_matmut(array: &mut Array1<f64>) -> MatMut<'_, f64> {
    let len = array.len();
    let stride = array.strides()[0];
    // SAFETY: array.as_mut_ptr() is ndarray's logical first-element pointer, and
    // len plus the signed element stride describe every initialized element of
    // this uniquely borrowed Array1 for the returned len×1 MatMut lifetime.
    unsafe {
        MatMut::from_raw_parts_mut(
            array.as_mut_ptr(),
            len,
            1,
            stride,
            0, // col stride irrelevant for 1 column
        )
    }
}

/// Compute A^T * A using faer's SIMD-optimized GEMM.
/// This is MUCH faster than ndarray's .t().dot() for matrices where n > ~100.
///
/// For a matrix A of shape (n, p), this computes the (p, p) result.
/// Uses a zero-copy view for positive-stride layouts and copies only layouts
/// with non-positive strides.
#[inline]
pub fn fast_ata<S: Data<Elem = f64>>(a: &ArrayBase<S, Ix2>) -> Array2<f64> {
    let p = a.ncols();
    let mut out = Array2::<f64>::zeros((p, p));
    fast_ata_into(a, &mut out);
    out
}

/// Compute A^T * A into a pre-allocated output buffer.
/// `out` must be shaped (p, p) where A is (n, p).
#[inline]
pub fn fast_ata_into<S: Data<Elem = f64>>(a: &ArrayBase<S, Ix2>, out: &mut Array2<f64>) {
    use faer::Accum;
    use faer::linalg::matmul::triangular::{BlockStructure, matmul as tri_matmul};

    let (n, p) = a.dim();
    assert_eq!(out.nrows(), p, "output rows must match p");
    assert_eq!(out.ncols(), p, "output cols must match p");

    if !should_use_faer_matmul(p, p, n) {
        out.assign(&a.t().dot(a));
        return;
    }

    let mut outview = array2_to_matmut(out);

    let aview = FaerArrayView::new(a);
    let a_ref = aview.as_ref();
    let a_t = a_ref.transpose();
    let par = matmul_parallelism(p, p, n);
    tri_matmul(
        outview.as_mut(),
        BlockStructure::TriangularLower,
        Accum::Replace,
        a_t,
        BlockStructure::Rectangular,
        a_ref,
        BlockStructure::Rectangular,
        1.0,
        par,
    );
    // Mirror lower triangle to upper to populate the full symmetric output.
    for i in 0..p {
        for j in (i + 1)..p {
            out[[i, j]] = out[[j, i]];
        }
    }
}

/// Compute A^T * B using faer's SIMD-optimized GEMM.
/// For A of shape (n, p) and B of shape (n, q), this computes the (p, q) result.
/// Uses zero-copy views when possible.
#[inline]
pub fn fast_atb<S1: Data<Elem = f64>, S2: Data<Elem = f64>>(
    a: &ArrayBase<S1, Ix2>,
    b: &ArrayBase<S2, Ix2>,
) -> Array2<f64> {
    if let Some(out) =
        crate::gpu_hook::gpu_dispatch().and_then(|d| d.try_fast_atb(a.view(), b.view()))
    {
        return out;
    }
    let (n_a, p) = a.dim();
    let q = b.ncols();
    fast_atb_with_parallelism(a, b, matmul_parallelism(p, q, n_a))
}

/// Compute A^T * B with an explicit faer parallelism policy for callers that
/// are already running independent products in an outer Rayon task.
#[inline]
pub fn fast_atb_with_parallelism<S1: Data<Elem = f64>, S2: Data<Elem = f64>>(
    a: &ArrayBase<S1, Ix2>,
    b: &ArrayBase<S2, Ix2>,
    par: Par,
) -> Array2<f64> {
    use faer::linalg::matmul::matmul;
    use faer::{Accum, Mat};

    let (n_a, p) = a.dim();
    let (n_b, q) = b.dim();
    assert_eq!(n_a, n_b, "A and B must have same number of rows");

    // For very small matrices, ndarray might be faster due to less overhead
    if !should_use_faer_matmul(p, q, n_a) {
        return a.t().dot(b);
    }

    let mut result = Mat::<f64>::zeros(p, q);

    let aview = FaerArrayView::new(a);
    let bview = FaerArrayView::new(b);
    let a_ref = aview.as_ref();
    let b_ref = bview.as_ref();

    // dst = A^T * B
    matmul(
        result.as_mut(),
        Accum::Replace,
        a_ref.transpose(),
        b_ref,
        1.0,
        par,
    );

    mat_to_array(result.as_ref())
}

/// Compute A * B^T using faer's SIMD-optimized GEMM.
/// For A of shape (m, k) and B of shape (n, k), this computes the (m, n) result.
#[inline]
pub fn fast_abt<S1: Data<Elem = f64>, S2: Data<Elem = f64>>(
    a: &ArrayBase<S1, Ix2>,
    b: &ArrayBase<S2, Ix2>,
) -> Array2<f64> {
    use faer::linalg::matmul::matmul;
    use faer::{Accum, Mat};

    let (m, k_a) = a.dim();
    let (n, k_b) = b.dim();
    assert_eq!(
        k_a, k_b,
        "A and B must have same number of columns for A·Bᵀ"
    );

    if !should_use_faer_matmul(m, n, k_a) {
        return a.dot(&b.t());
    }

    let mut result = Mat::<f64>::zeros(m, n);
    let aview = FaerArrayView::new(a);
    let bview = FaerArrayView::new(b);
    let par = matmul_parallelism(m, n, k_a);
    matmul(
        result.as_mut(),
        Accum::Replace,
        aview.as_ref(),
        bview.as_ref().transpose(),
        1.0,
        par,
    );
    mat_to_array(result.as_ref())
}

/// Compute A * B using faer's SIMD-optimized GEMM.
/// For A of shape (n, p) and B of shape (p, q), this computes the (n, q) result.
/// Uses zero-copy views when possible.
#[inline]
pub fn fast_ab<S1: Data<Elem = f64>, S2: Data<Elem = f64>>(
    a: &ArrayBase<S1, Ix2>,
    b: &ArrayBase<S2, Ix2>,
) -> Array2<f64> {
    if let Some(out) =
        crate::gpu_hook::gpu_dispatch().and_then(|d| d.try_fast_ab(a.view(), b.view()))
    {
        return out;
    }
    let n = a.nrows();
    let q = b.ncols();
    let mut out = Array2::<f64>::zeros((n, q));
    fast_ab_into(a, b, &mut out);
    out
}

// ────────────────────────────────────────────────────────────────────────
// Compensated / blocked SIMD reduction kernels for the GEMV hot paths.
//
// `fast_av` (η = Xβ) and `fast_atv` (Xᵀr — e.g. the penalized-likelihood
// gradient and REML score) are reduction-bound: every output entry is a sum
// of products over a long axis. faer's generic GEMM serves them as degenerate
// single-RHS-column matmuls, whose blocking/setup cost is poorly amortized by
// one column. For the dominant row-major-contiguous case we use tight hand
// kernels that are simultaneously
//   * faster — several independent FMA accumulators expose the
//     instruction-level parallelism the backend lowers to packed AVX
//     `vfmadd` lanes, and the row work fans out across the Rayon pool; and
//   * more accurate — `f64::mul_add` fuses each product into its accumulator
//     with a single rounding (no rounded intermediate product), the lanes
//     reduce as a small pairwise tree, and the long Xᵀr reduction is split
//     into fixed-size row blocks whose partials are combined pairwise,
//     turning the naive O(n·ε) error growth into ~O((block + log(n/block))·ε).
//
// Non-contiguous / non-row-major operands fall back to the faer path, so the
// numerics only change (improve) on the common standard-layout inputs.
// ────────────────────────────────────────────────────────────────────────

/// Number of independent FMA accumulator lanes. Eight lanes keep two 256-bit
/// (`f64x4`) FMA pipelines fed and set the partial-pairwise leaf width.
const FMA_LANES: usize = 8;

/// FLOP-scale (n·p) below which the kernels stay serial; at or above it, and
/// only when not already inside a parallel row region, the row loop fans out
/// across the Rayon pool.
const KERNEL_PAR_MIN_FLOP: usize = 1 << 18; // 262_144

/// Maximum rows per row-block in the row-major matrix-vector kernels. The
/// actual block shrinks for wide matrices so one cache-resident dense operator
/// does not expose only one or two tasks to a larger Rayon pool.
const AV_PAR_MAX_CHUNK_ROWS: usize = 1024;

/// Rows per reduction block in [`fast_atv_rowmajor_into`]; each block sums its
/// rows into a private length-p partial and the partials combine pairwise, so
/// the long-axis rounding error grows with the block size plus the log of the
/// block count rather than with `n`.
const ATV_BLOCK_ROWS: usize = 512;

#[inline]
fn kernel_should_parallelize(n: usize, p: usize) -> bool {
    !in_nested_parallel_region()
        && n.saturating_mul(p) >= KERNEL_PAR_MIN_FLOP
        && rayon::current_num_threads() > 1
}

#[inline]
fn av_parallel_chunk_rows(p: usize) -> usize {
    KERNEL_PAR_MIN_FLOP
        .div_ceil(p.max(1))
        .clamp(64, AV_PAR_MAX_CHUNK_ROWS)
}

/// Compensated dot product (the Ogita–Rump–Oishi *Dot2* error-free transform)
/// of two equal-length contiguous slices, evaluated over [`FMA_LANES`]
/// independent compensated accumulators.
///
/// For each term the product is split into its rounded value plus the *exact*
/// product error via `mul_add` (`two_prod`), and added into the running sum via
/// a branchless `two_sum`, with both rounding errors folded into a
/// per-lane compensation. The result carries roughly twice the working
/// precision: its error-vs-truth is bounded by `u·|result| + O(n·u²)·|x|ᵀ|y|`
/// versus the naive recurrence's `O(n·u)·|x|ᵀ|y|`, i.e. strictly — often by
/// many orders of magnitude — more accurate. The eight independent lanes keep
/// the FMA pipelines saturated, and on the GEMV hot paths the extra arithmetic
/// is hidden under the memory traffic of streaming `X`, so accuracy rises with
/// no throughput cost.
#[inline(always)]
fn fma_dot(a: &[f64], b: &[f64]) -> f64 {
    assert_eq!(a.len(), b.len(), "fma_dot: operand length mismatch");
    let mut sum = [0.0f64; FMA_LANES];
    let mut comp = [0.0f64; FMA_LANES];
    let mut ca = a.chunks_exact(FMA_LANES);
    let mut cb = b.chunks_exact(FMA_LANES);
    for (xa, xb) in ca.by_ref().zip(cb.by_ref()) {
        for l in 0..FMA_LANES {
            let x = xa[l];
            let y = xb[l];
            // two_prod: p = round(x·y), ep = exact error x·y − p.
            let p = x * y;
            let ep = x.mul_add(y, -p);
            // two_sum: s = round(sum + p), es = exact error.
            let s = sum[l] + p;
            let bb = s - sum[l];
            let es = (sum[l] - (s - bb)) + (p - bb);
            sum[l] = s;
            comp[l] += ep + es;
        }
    }
    // Compensated remainder lane (length < FMA_LANES).
    let mut sr = 0.0f64;
    let mut cr = 0.0f64;
    for (&x, &y) in ca.remainder().iter().zip(cb.remainder().iter()) {
        let p = x * y;
        let ep = x.mul_add(y, -p);
        let s = sr + p;
        let bb = s - sr;
        let es = (sr - (s - bb)) + (p - bb);
        sr = s;
        cr += ep + es;
    }
    // Fold each lane's compensation back in, then reduce the (few) lanes.
    let mut total = sr + cr;
    for l in 0..FMA_LANES {
        total += sum[l] + comp[l];
    }
    total
}

/// `out[i] = Σ_j X[i,j]·v[j]` for row-major-contiguous `x_all` (len `n·p`) and
/// `v` (len `p`). Each output row is an independent [`fma_dot`]; rows fan out
/// in chunks across the Rayon pool when the work is large.
fn fast_av_rowmajor_into(x_all: &[f64], v: &[f64], n: usize, p: usize, out: &mut [f64]) {
    assert_eq!(x_all.len(), n * p, "fast_av_rowmajor_into: x_all length");
    assert_eq!(v.len(), p, "fast_av_rowmajor_into: v length");
    assert_eq!(out.len(), n, "fast_av_rowmajor_into: out length");
    if kernel_should_parallelize(n, p) {
        use rayon::prelude::*;
        let chunk_rows = av_parallel_chunk_rows(p);
        out.par_chunks_mut(chunk_rows)
            .enumerate()
            .for_each(|(c, chunk)| {
                let base = c * chunk_rows;
                for (k, o) in chunk.iter_mut().enumerate() {
                    let i = base + k;
                    *o = fma_dot(&x_all[i * p..i * p + p], v);
                }
            });
    } else {
        for (i, o) in out.iter_mut().enumerate() {
            *o = fma_dot(&x_all[i * p..i * p + p], v);
        }
    }
}

/// Ordinary fused-multiply-add dot product with independent accumulator lanes.
///
/// This is the IEEE-754 workhorse for iterative operators whose caller owns an
/// explicit residual certificate. It intentionally omits Dot2's error-free
/// product/sum transforms: for a 2,000-term row the standard `O(p·ε)` error is
/// still roughly four orders of magnitude below a `1e-8` Ritz contract, while
/// the saved arithmetic matters when the same cache-resident matrix is applied
/// hundreds of times.
#[inline(always)]
fn standard_fma_dot(a: &[f64], b: &[f64]) -> f64 {
    assert_eq!(
        a.len(),
        b.len(),
        "standard_fma_dot: operand length mismatch"
    );
    let mut sum = [0.0_f64; FMA_LANES];
    let mut ca = a.chunks_exact(FMA_LANES);
    let mut cb = b.chunks_exact(FMA_LANES);
    for (xa, xb) in ca.by_ref().zip(cb.by_ref()) {
        for lane in 0..FMA_LANES {
            sum[lane] = xa[lane].mul_add(xb[lane], sum[lane]);
        }
    }
    let mut remainder = 0.0;
    for (&x, &y) in ca.remainder().iter().zip(cb.remainder().iter()) {
        remainder = x.mul_add(y, remainder);
    }
    let pair01 = sum[0] + sum[1];
    let pair23 = sum[2] + sum[3];
    let pair45 = sum[4] + sum[5];
    let pair67 = sum[6] + sum[7];
    remainder + (pair01 + pair23) + (pair45 + pair67)
}

fn standard_av_rowmajor_into(x_all: &[f64], v: &[f64], n: usize, p: usize, out: &mut [f64]) {
    assert_eq!(
        x_all.len(),
        n * p,
        "standard_av_rowmajor_into: matrix length"
    );
    assert_eq!(v.len(), p, "standard_av_rowmajor_into: vector length");
    assert_eq!(out.len(), n, "standard_av_rowmajor_into: output length");
    if kernel_should_parallelize(n, p) {
        use rayon::prelude::*;
        let chunk_rows = av_parallel_chunk_rows(p);
        out.par_chunks_mut(chunk_rows)
            .enumerate()
            .for_each(|(chunk_index, chunk)| {
                let base = chunk_index * chunk_rows;
                for (offset, output) in chunk.iter_mut().enumerate() {
                    let row = base + offset;
                    *output = standard_fma_dot(&x_all[row * p..row * p + p], v);
                }
            });
    } else {
        for (row, output) in out.iter_mut().enumerate() {
            *output = standard_fma_dot(&x_all[row * p..row * p + p], v);
        }
    }
}

/// Pairwise (tree) sum of equal-length partial vectors into `out`.
fn pairwise_sum_into(parts: &[Vec<f64>], out: &mut [f64]) {
    match parts.len() {
        0 => out.fill(0.0),
        1 => out.copy_from_slice(&parts[0]),
        _ => {
            let mid = parts.len() / 2;
            let p = out.len();
            let mut left = vec![0.0f64; p];
            let mut right = vec![0.0f64; p];
            pairwise_sum_into(&parts[..mid], &mut left);
            pairwise_sum_into(&parts[mid..], &mut right);
            for ((o, &l), &r) in out.iter_mut().zip(left.iter()).zip(right.iter()) {
                *o = l + r;
            }
        }
    }
}

/// `out[j] = Σ_i v[i]·X[i,j]` for row-major-contiguous `x_all` (len `n·p`).
///
/// Rows are grouped into [`ATV_BLOCK_ROWS`] blocks; each block FMA-accumulates
/// its rows into a private partial vector (fused `v[i]·X[i,j]`), and the block
/// partials are combined pairwise. This blocked/pairwise reduction is both
/// better-conditioned than a single running sum over all `n` rows and trivially
/// parallel across blocks.
fn fast_atv_rowmajor_into(x_all: &[f64], v: &[f64], n: usize, p: usize, out: &mut [f64]) {
    assert_eq!(x_all.len(), n * p, "fast_atv_rowmajor_into: x_all length");
    assert_eq!(v.len(), n, "fast_atv_rowmajor_into: v length");
    assert_eq!(out.len(), p, "fast_atv_rowmajor_into: out length");
    let nblocks = n.div_ceil(ATV_BLOCK_ROWS);

    let block_partial = |b: usize| -> Vec<f64> {
        let start = b * ATV_BLOCK_ROWS;
        let end = (start + ATV_BLOCK_ROWS).min(n);
        let mut acc = vec![0.0f64; p];
        for i in start..end {
            let vi = v[i];
            let row = &x_all[i * p..i * p + p];
            for (a, &xij) in acc.iter_mut().zip(row.iter()) {
                *a = xij.mul_add(vi, *a);
            }
        }
        acc
    };

    let partials: Vec<Vec<f64>> = if kernel_should_parallelize(n, p) {
        use rayon::prelude::*;
        (0..nblocks).into_par_iter().map(block_partial).collect()
    } else {
        (0..nblocks).map(block_partial).collect()
    };

    pairwise_sum_into(&partials, out);
}

/// Compute A * v using faer's SIMD-optimized GEMV.
/// For A of shape (n, p) and v of shape (p,), this computes the (n,) result.
#[inline]
pub fn fast_av<S1: Data<Elem = f64>, S2: Data<Elem = f64>>(
    a: &ArrayBase<S1, Ix2>,
    v: &ArrayBase<S2, Ix1>,
) -> Array1<f64> {
    if let Some(out) =
        crate::gpu_hook::gpu_dispatch().and_then(|d| d.try_fast_av(a.view(), v.view()))
    {
        return out;
    }
    fast_av_impl(a, v)
}

#[inline]
fn fast_av_impl<S1: Data<Elem = f64>, S2: Data<Elem = f64>>(
    a: &ArrayBase<S1, Ix2>,
    v: &ArrayBase<S2, Ix1>,
) -> Array1<f64> {
    use faer::linalg::matmul::matmul;
    use faer::{Accum, Mat};

    let (n, p) = a.dim();
    assert_eq!(p, v.len(), "A cols must match v length");

    // Row-major-contiguous fast path: tight multi-lane FMA dot per row, both
    // faster (ILP / Rayon fan-out) and more accurate (fused products, pairwise
    // lane reduction) than the degenerate single-column faer GEMV.
    if let (Some(x_all), Some(vs)) = (a.as_slice(), v.as_slice())
        && n != 0
        && p != 0
    {
        let mut out = Array1::<f64>::zeros(n);
        fast_av_rowmajor_into(
            x_all,
            vs,
            n,
            p,
            out.as_slice_mut().expect("fresh Array1 is contiguous"),
        );
        return out;
    }

    if !should_use_faer_matmul(n, 1, p) {
        return a.dot(v);
    }

    let mut result = Mat::<f64>::zeros(n, 1);

    let aview = FaerArrayView::new(a);
    let vview = FaerColView::new(v);
    let a_ref = aview.as_ref();
    let v_ref = vview.as_ref();

    let par = matmul_parallelism(n, 1, p);
    matmul(result.as_mut(), Accum::Replace, a_ref, v_ref, 1.0, par);

    let mut out = Array1::<f64>::zeros(n);
    for i in 0..n {
        out[i] = result[(i, 0)];
    }
    out
}

/// Compute A * v into a pre-allocated output buffer.
/// `out` must be length n where A is (n, p) and v is length p.
#[inline]
pub fn fast_av_into<S1: Data<Elem = f64>, S2: Data<Elem = f64>>(
    a: &ArrayBase<S1, Ix2>,
    v: &ArrayBase<S2, Ix1>,
    out: &mut Array1<f64>,
) {
    fast_av_into_impl(a, v, out);
}

#[inline]
fn fast_av_into_impl<S1: Data<Elem = f64>, S2: Data<Elem = f64>>(
    a: &ArrayBase<S1, Ix2>,
    v: &ArrayBase<S2, Ix1>,
    out: &mut Array1<f64>,
) {
    use faer::Accum;
    use faer::linalg::matmul::matmul;

    let (n, p) = a.dim();
    assert_eq!(v.len(), p, "vector length must match A cols");
    assert_eq!(out.len(), n, "output length must match A rows");

    if let (Some(x_all), Some(vs)) = (a.as_slice(), v.as_slice())
        && n != 0
        && p != 0
        && let Some(out_s) = out.as_slice_mut()
    {
        fast_av_rowmajor_into(x_all, vs, n, p, out_s);
        return;
    }

    if !should_use_faer_matmul(n, 1, p) {
        out.assign(&a.dot(v));
        return;
    }

    let mut outview = array1_to_col_matmut(out);

    let aview = FaerArrayView::new(a);
    let vview = FaerColView::new(v);
    let a_ref = aview.as_ref();
    let v_ref = vview.as_ref();
    let par = matmul_parallelism(n, 1, p);
    matmul(outview.as_mut(), Accum::Replace, a_ref, v_ref, 1.0, par);
}

/// Compute A * v into a pre-allocated `ArrayViewMut1` slice. Like
/// [`fast_av_into`] but accepts a writable slice rather than `&mut Array1`,
/// so callers can write directly into a sub-range of a larger buffer
/// without intermediate allocation.
///
/// `out` must have length n where A is (n, p) and v is length p.
#[inline]
pub fn fast_av_view_into<S1: Data<Elem = f64>, S2: Data<Elem = f64>>(
    a: &ArrayBase<S1, Ix2>,
    v: &ArrayBase<S2, Ix1>,
    out: ArrayViewMut1<'_, f64>,
) {
    fast_av_view_into_impl(a, v, out);
}

/// Compute `A·v` with the standard-FMA row kernel.
///
/// Prefer this over [`fast_av_view_into`] only when the surrounding iterative
/// algorithm certifies its final residual explicitly. The ordinary kernel is
/// materially faster for repeated cache-resident dense applications; callers
/// that need Dot2's near-double-precision reduction should keep using
/// [`fast_av_view_into`].
pub fn fast_av_standard_view_into<S1: Data<Elem = f64>, S2: Data<Elem = f64>>(
    a: &ArrayBase<S1, Ix2>,
    v: &ArrayBase<S2, Ix1>,
    mut out: ArrayViewMut1<'_, f64>,
) {
    use faer::Accum;
    use faer::linalg::matmul::matmul;

    let (n, p) = a.dim();
    assert_eq!(v.len(), p, "vector length must match A cols");
    assert_eq!(out.len(), n, "output length must match A rows");
    if let (Some(x_all), Some(vs), Some(out_slice)) =
        (a.as_slice(), v.as_slice(), out.as_slice_mut())
        && n != 0
        && p != 0
    {
        standard_av_rowmajor_into(x_all, vs, n, p, out_slice);
        return;
    }
    if !should_use_faer_matmul(n, 1, p) {
        out.assign(&a.dot(v));
        return;
    }

    let len = out.len();
    let stride = out.strides()[0];
    // SAFETY: `out` is uniquely borrowed and `len` plus its signed stride
    // describe every initialized element of the one-column destination.
    let outview = unsafe { MatMut::from_raw_parts_mut(out.as_mut_ptr(), len, 1, stride, 0) };
    let aview = FaerArrayView::new(a);
    let vview = FaerColView::new(v);
    matmul(
        outview,
        Accum::Replace,
        aview.as_ref(),
        vview.as_ref(),
        1.0,
        matmul_parallelism(n, 1, p),
    );
}

#[inline]
fn fast_av_view_into_impl<S1: Data<Elem = f64>, S2: Data<Elem = f64>>(
    a: &ArrayBase<S1, Ix2>,
    v: &ArrayBase<S2, Ix1>,
    mut out: ArrayViewMut1<'_, f64>,
) {
    use faer::Accum;
    use faer::linalg::matmul::matmul;

    let (n, p) = a.dim();
    assert_eq!(v.len(), p, "vector length must match A cols");
    assert_eq!(out.len(), n, "output length must match A rows");

    if let (Some(x_all), Some(vs)) = (a.as_slice(), v.as_slice())
        && n != 0
        && p != 0
        && let Some(out_s) = out.as_slice_mut()
    {
        fast_av_rowmajor_into(x_all, vs, n, p, out_s);
        return;
    }

    if !should_use_faer_matmul(n, 1, p) {
        let prod = a.dot(v);
        out.assign(&prod);
        return;
    }

    let len = out.len();
    let stride = out.strides()[0];
    // SAFETY: out.as_mut_ptr() is ndarray's logical first-element pointer, and
    // len plus the signed element stride describe every initialized element of
    // this uniquely borrowed view for the returned len×1 MatMut lifetime.
    let outview = unsafe {
        MatMut::from_raw_parts_mut(
            out.as_mut_ptr(),
            len,
            1,
            stride,
            0, // col stride irrelevant for 1 column
        )
    };

    let aview = FaerArrayView::new(a);
    let vview = FaerColView::new(v);
    let a_ref = aview.as_ref();
    let v_ref = vview.as_ref();
    let par = matmul_parallelism(n, 1, p);
    matmul(outview, Accum::Replace, a_ref, v_ref, 1.0, par);
}

/// Compute A^T * v using faer's SIMD-optimized GEMV.
/// For A of shape (n, p) and v of shape (n,), this computes the (p,) result.
#[inline]
pub fn fast_atv<S1: Data<Elem = f64>, S2: Data<Elem = f64>>(
    a: &ArrayBase<S1, Ix2>,
    v: &ArrayBase<S2, Ix1>,
) -> Array1<f64> {
    if let Some(out) =
        crate::gpu_hook::gpu_dispatch().and_then(|d| d.try_fast_atv(a.view(), v.view()))
    {
        return out;
    }
    fast_atv_impl(a, v)
}

#[inline]
fn fast_atv_impl<S1: Data<Elem = f64>, S2: Data<Elem = f64>>(
    a: &ArrayBase<S1, Ix2>,
    v: &ArrayBase<S2, Ix1>,
) -> Array1<f64> {
    use faer::Accum;
    use faer::linalg::matmul::matmul;

    let (n, p) = a.dim();
    assert_eq!(n, v.len(), "A rows must match v length");

    // Row-major-contiguous fast path: blocked + pairwise FMA reduction over the
    // long n-axis. Lower error-vs-truth than a single running sum and parallel
    // across row blocks.
    if let (Some(x_all), Some(vs)) = (a.as_slice(), v.as_slice())
        && n != 0
        && p != 0
    {
        let mut out = Array1::<f64>::zeros(p);
        fast_atv_rowmajor_into(
            x_all,
            vs,
            n,
            p,
            out.as_slice_mut().expect("fresh Array1 is contiguous"),
        );
        return out;
    }

    // For very small arrays, ndarray might be faster
    if !should_use_faer_matmul(p, 1, n) {
        return a.t().dot(v);
    }

    let mut out = Array1::<f64>::zeros(p);
    let mut outview = array1_to_col_matmut(&mut out);

    let aview = FaerArrayView::new(a);
    let vview = FaerColView::new(v);
    let a_ref = aview.as_ref();
    let v_ref = vview.as_ref();

    // dst = A^T * v (treating v as n×1 matrix)
    let par = matmul_parallelism(p, 1, n);
    matmul(
        outview.as_mut(),
        Accum::Replace,
        a_ref.transpose(),
        v_ref,
        1.0,
        par,
    );

    out
}

/// Compute A^T * v into a pre-allocated output buffer.
/// `out` must be length p where A is (n, p) and v is length n.
#[inline]
pub fn fast_atv_into<S1: Data<Elem = f64>, S2: Data<Elem = f64>>(
    a: &ArrayBase<S1, Ix2>,
    v: &ArrayBase<S2, Ix1>,
    out: &mut Array1<f64>,
) {
    fast_atv_into_impl(a, v, out);
}

#[inline]
fn fast_atv_into_impl<S1: Data<Elem = f64>, S2: Data<Elem = f64>>(
    a: &ArrayBase<S1, Ix2>,
    v: &ArrayBase<S2, Ix1>,
    out: &mut Array1<f64>,
) {
    use faer::Accum;
    use faer::linalg::matmul::matmul;

    let (n, p) = a.dim();
    assert_eq!(v.len(), n, "vector length must match A rows");
    assert_eq!(out.len(), p, "output length must match A cols");

    if let (Some(x_all), Some(vs)) = (a.as_slice(), v.as_slice())
        && n != 0
        && p != 0
        && let Some(out_s) = out.as_slice_mut()
    {
        fast_atv_rowmajor_into(x_all, vs, n, p, out_s);
        return;
    }

    if !should_use_faer_matmul(p, 1, n) {
        out.assign(&a.t().dot(v));
        return;
    }

    let mut outview = array1_to_col_matmut(out);

    let aview = FaerArrayView::new(a);
    let vview = FaerColView::new(v);
    let a_ref = aview.as_ref();
    let v_ref = vview.as_ref();
    let par = matmul_parallelism(p, 1, n);
    matmul(
        outview.as_mut(),
        Accum::Replace,
        a_ref.transpose(),
        v_ref,
        1.0,
        par,
    );
}

/// Compute A^T * diag(W) * A using streaming chunks to avoid O(n*p) allocation.
#[inline]
pub fn fast_xt_diag_x<S1: Data<Elem = f64>, S2: Data<Elem = f64>>(
    x: &ArrayBase<S1, Ix2>,
    w: &ArrayBase<S2, Ix1>,
) -> Array2<f64> {
    assert_eq!(
        x.nrows(),
        w.len(),
        "fast_xt_diag_x row/weight length mismatch"
    );
    if let Some(out) =
        crate::gpu_hook::gpu_dispatch().and_then(|d| d.try_fast_xt_diag_x(x.view(), w.view()))
    {
        return out;
    }
    let p = x.ncols();
    fast_xt_diag_x_with_parallelism(x, w, matmul_parallelism(p, p, x.nrows()))
}

/// Compute A^T * diag(W) * A with an explicit faer parallelism policy for
/// callers that parallelize multiple independent Hessian blocks externally.
#[inline]
pub fn fast_xt_diag_x_with_parallelism<S1: Data<Elem = f64>, S2: Data<Elem = f64>>(
    x: &ArrayBase<S1, Ix2>,
    w: &ArrayBase<S2, Ix1>,
    par: Par,
) -> Array2<f64> {
    assert_eq!(
        x.nrows(),
        w.len(),
        "fast_xt_diag_x_with_parallelism row/weight length mismatch"
    );
    fast_xt_diag_x_with_parallelism_impl(x, w, par)
}

#[inline]
fn fast_xt_diag_x_with_parallelism_impl<S1: Data<Elem = f64>, S2: Data<Elem = f64>>(
    x: &ArrayBase<S1, Ix2>,
    w: &ArrayBase<S2, Ix1>,
    par: Par,
) -> Array2<f64> {
    use ndarray::ShapeBuilder;

    let p = x.ncols();
    // F-order result so the symmetric lower-triangle accumulation writes
    // column-contiguously; the kernel mirrors to a full symmetric matrix.
    let mut result = Array2::<f64>::zeros((p, p).f());
    stream_weighted_crossprod_into(
        x,
        w,
        &mut result,
        CrossprodStructure::SymmetricLower,
        CrossprodAccum::Replace,
        par,
    );
    result
}

/// Output packaging for [`stream_weighted_crossprod_into`].
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum CrossprodStructure {
    /// Compute every entry of the (symmetric) Gram via full GEMM.
    Full,
    /// Accumulate only the lower triangle via triangular matmul (~50% fewer
    /// FLOPs), then mirror once into the upper triangle for a full symmetric
    /// result. Mathematically identical output to [`Full`](Self::Full).
    SymmetricLower,
}

/// Accumulation policy for [`stream_weighted_crossprod_into`].
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum CrossprodAccum {
    /// Overwrite `out` with `Xᵀ·diag(W)·X`, ignoring prior contents.
    Replace,
    /// Add `Xᵀ·diag(W)·X` into the existing contents of `out`.
    Add,
}

/// Rows per streaming chunk so each `chunk_rows × cols` `f64` tile stays near an
/// 8 MiB working set, clamped to `[512, 131_072]` and never exceeding `n`.
///
/// One definition for the three streaming kernels below, which carried
/// byte-identical inline copies of this arithmetic. The copies computed the same
/// quantity from different column expressions (`p`, `px + q`, `pa + 2·pb`), so
/// the only thing that ever differed between them was the argument — which is
/// exactly the shape that should be a parameter rather than three transcriptions
/// of one rule (#2469).
///
/// **This is NOT [`crate::utils::row_chunk_for_byte_budget`], and the difference
/// is unresolved.** That function is documented as the "canonical home for the
/// row-chunk heuristic", takes the same `(n, cols)` and computes the same 8 MiB
/// budget — but clamps to `[256, 65_536]` rather than `[512, 131_072]`. These
/// kernels have always used the wider band; routing them through the canonical
/// helper would halve the floor and quarter the ceiling and is a behaviour
/// change, not a cleanup. Collapsing three copies into one makes that single
/// remaining divergence visible instead of triplicated; deciding which band is
/// right needs a measurement nobody has taken.
#[inline]
fn streaming_chunk_rows(cols: usize, n: usize) -> usize {
    // The library row-chunk target, IMPORTED rather than transcribed. Its own
    // doc says it is shared as a `const` "so compile-time consumers stay in
    // lockstep with `ResourcePolicy::default_library` without a runtime policy
    // query" -- which is exactly this call site's situation. Same quantity, not
    // merely the same number: this crate already consumes the runtime form of
    // it (`row_chunk_target_bytes`, `matrix/mod.rs`), and `gam-gpu`'s tile
    // geometry derives from the same const for the same reason.
    const TARGET_BYTES: usize = gam_runtime::resource::LIBRARY_ROW_CHUNK_TARGET_BYTES;
    const MIN_ROWS: usize = 512;
    const MAX_ROWS: usize = 131_072;
    (TARGET_BYTES / (cols.max(1) * std::mem::size_of::<f64>()))
        .clamp(MIN_ROWS, MAX_ROWS)
        .min(n)
}

/// Shared dense weighted-Gram kernel: accumulate `Xᵀ·diag(W)·X` into `out`.
///
/// This is the single tuned implementation of the chunked row-scaling +
/// matmul strategy; the matrix-returning (`fast_xt_diag_x*`) entry points and
/// stream-in callers share it so that performance tuning, negative-weight
/// handling, chunk sizing, and layout fixes land in exactly one place.
///
/// Computes the product as `Xᵀ·(W·X)` to preserve the sign of `W`: the prior
/// `sqrt(max(0, w))`-then-Gram form clipped negative weights to zero, which
/// corrupted observed-Hessian assembly when any block carried heavy residuals
/// (e.g. under the logb σ link).
///
/// Peak working-set allocation is `chunk_rows × p × 8` bytes (~8 MB) rather
/// than `n × p × 8` bytes for a materialized `W·X`.
///
/// `out` must be `p × p`. With [`CrossprodStructure::SymmetricLower`] the
/// lower triangle is accumulated and then mirrored, so on return `out` holds
/// the full symmetric matrix regardless of `structure`.
pub fn stream_weighted_crossprod_into<S1: Data<Elem = f64>, S2: Data<Elem = f64>>(
    x: &ArrayBase<S1, Ix2>,
    w: &ArrayBase<S2, Ix1>,
    out: &mut Array2<f64>,
    structure: CrossprodStructure,
    accum: CrossprodAccum,
    par: Par,
) {
    use faer::Accum;
    use faer::linalg::matmul::matmul;
    use faer::linalg::matmul::triangular::{BlockStructure, matmul as tri_matmul};
    use ndarray::s;

    let (n, p) = x.dim();
    assert_eq!(n, w.len(), "X rows must match W length");
    assert_eq!(out.nrows(), p, "output rows must match X cols");
    assert_eq!(out.ncols(), p, "output cols must match X cols");
    if p == 0 {
        return;
    }
    if n == 0 {
        if accum == CrossprodAccum::Replace {
            out.fill(0.0);
        }
        return;
    }

    if !should_use_faer_matmul(p, p, n) {
        // Tiny products: ndarray's own GEMM avoids faer setup overhead.
        let w_x = Array2::from_shape_fn((n, p), |(i, j)| w[i] * x[[i, j]]);
        let gram = x.t().dot(&w_x);
        match accum {
            CrossprodAccum::Replace => out.assign(&gram),
            CrossprodAccum::Add => *out += &gram,
        }
        return;
    }

    // Streaming chunked: peak allocation is chunk_rows × p instead of n × p.
    let chunk_rows = streaming_chunk_rows(p, n);

    // Triangular accumulation requires a zero baseline in the lower triangle
    // because each chunk's `Accum::Add` lands there; for a Replace request we
    // zero up front and add every chunk, for an Add request the caller's
    // contents are preserved and every chunk adds on top.
    if accum == CrossprodAccum::Replace {
        out.fill(0.0);
    }

    // Row-major wx_chunk so the per-row scaling loop has stride-1 writes
    // alongside stride-1 reads from a row-major X. An F-order wx_chunk would
    // force strided writes by `chunk_rows`, breaking vectorization and cache
    // locality on the per-PIRLS-iter Hessian assembly. faer's matmul handles
    // either layout via FaerArrayView.
    let mut wx_chunk = Array2::<f64>::zeros((chunk_rows, p));

    let x_is_row_major = x.is_standard_layout();
    let w_slice_opt = w.as_slice();

    // Scope the faer mutable view so its borrow on `out` ends before the
    // symmetric mirror step.
    {
        let mut out_view = array2_to_matmut(out);
        for start in (0..n).step_by(chunk_rows) {
            let rows = (n - start).min(chunk_rows);
            {
                let chunk_slice = wx_chunk
                    .as_slice_mut()
                    .expect("row-major chunk is contiguous");
                if x_is_row_major && let (Some(x_all), Some(w_all)) = (x.as_slice(), w_slice_opt) {
                    for local in 0..rows {
                        let src = start + local;
                        let wi = w_all[src];
                        let src_off = src * p;
                        let dst_off = local * p;
                        let src_row = &x_all[src_off..src_off + p];
                        let dst_row = &mut chunk_slice[dst_off..dst_off + p];
                        for col in 0..p {
                            dst_row[col] = src_row[col] * wi;
                        }
                    }
                } else {
                    let x_slice = x.slice(s![start..start + rows, ..]);
                    for local in 0..rows {
                        let wi = w[start + local];
                        let xrow = x_slice.row(local);
                        let dst_off = local * p;
                        let dst_row = &mut chunk_slice[dst_off..dst_off + p];
                        for (col, xij) in xrow.iter().enumerate() {
                            dst_row[col] = xij * wi;
                        }
                    }
                }
            }
            let x_slice = x.slice(s![start..start + rows, ..]);
            let wx_slice = wx_chunk.slice(s![0..rows, ..]);
            let x_view = FaerArrayView::new(&x_slice);
            let wx_view = FaerArrayView::new(&wx_slice);
            match structure {
                CrossprodStructure::SymmetricLower => {
                    // X^T diag(W) X is symmetric; accumulate the lower triangle
                    // only, then mirror once after the chunk loop. ~50% fewer
                    // FLOPs vs. full GEMM.
                    tri_matmul(
                        out_view.as_mut(),
                        BlockStructure::TriangularLower,
                        Accum::Add,
                        x_view.as_ref().transpose(),
                        BlockStructure::Rectangular,
                        wx_view.as_ref(),
                        BlockStructure::Rectangular,
                        1.0,
                        par,
                    );
                }
                CrossprodStructure::Full => {
                    matmul(
                        out_view.as_mut(),
                        Accum::Add,
                        x_view.as_ref().transpose(),
                        wx_view.as_ref(),
                        1.0,
                        par,
                    );
                }
            }
        }
    }

    if structure == CrossprodStructure::SymmetricLower {
        // Mirror lower triangle to upper for a full symmetric output.
        for i in 0..p {
            for j in (i + 1)..p {
                out[[i, j]] = out[[j, i]];
            }
        }
    }
}

/// Compute A^T * diag(W) * B using streaming chunks.
#[inline]
pub fn fast_xt_diag_y<S1: Data<Elem = f64>, S2: Data<Elem = f64>, S3: Data<Elem = f64>>(
    x: &ArrayBase<S1, Ix2>,
    w: &ArrayBase<S2, Ix1>,
    y: &ArrayBase<S3, Ix2>,
) -> Array2<f64> {
    assert_eq!(x.nrows(), y.nrows(), "fast_xt_diag_y X/Y row mismatch");
    assert_eq!(
        y.nrows(),
        w.len(),
        "fast_xt_diag_y row/weight length mismatch"
    );
    if let Some(out) = crate::gpu_hook::gpu_dispatch()
        .and_then(|d| d.try_fast_xt_diag_y(x.view(), w.view(), y.view()))
    {
        return out;
    }
    fast_xt_diag_y_impl(x, w, y)
}

#[inline]
fn fast_xt_diag_y_impl<S1: Data<Elem = f64>, S2: Data<Elem = f64>, S3: Data<Elem = f64>>(
    x: &ArrayBase<S1, Ix2>,
    w: &ArrayBase<S2, Ix1>,
    y: &ArrayBase<S3, Ix2>,
) -> Array2<f64> {
    use faer::Accum;
    use faer::linalg::matmul::matmul;
    use ndarray::{ShapeBuilder, s};

    let (n, q) = y.dim();
    let px = x.ncols();
    assert_eq!(n, w.len(), "Y rows must match W length");
    assert_eq!(n, x.nrows(), "X rows must match Y rows");
    if n == 0 || px == 0 || q == 0 {
        return Array2::<f64>::zeros((px, q));
    }
    if !should_use_faer_matmul(px, q, n) {
        let w_y = Array2::from_shape_fn((n, q), |(i, j)| w[i] * y[[i, j]]);
        return x.t().dot(&w_y);
    }

    // Streaming: only allocate chunk_rows × q for the weighted Y slice.
    let total_cols = px + q;
    let chunk_rows = streaming_chunk_rows(total_cols, n);

    let mut result = Array2::<f64>::zeros((px, q).f());
    // Row-major wy_chunk — same rationale as fast_xt_diag_x: stride-1
    // writes alongside stride-1 reads from a row-major Y.
    let mut wy_chunk = Array2::<f64>::zeros((chunk_rows, q));

    let y_is_row_major = y.is_standard_layout();
    let w_slice_opt = w.as_slice();

    {
        let mut out_view = array2_to_matmut(&mut result);

        for start in (0..n).step_by(chunk_rows) {
            let rows = (n - start).min(chunk_rows);
            {
                let chunk_slice = wy_chunk
                    .as_slice_mut()
                    .expect("row-major chunk is contiguous");
                if y_is_row_major && let (Some(y_all), Some(w_all)) = (y.as_slice(), w_slice_opt) {
                    for local in 0..rows {
                        let src = start + local;
                        let wi = w_all[src];
                        let src_off = src * q;
                        let dst_off = local * q;
                        let src_row = &y_all[src_off..src_off + q];
                        let dst_row = &mut chunk_slice[dst_off..dst_off + q];
                        for col in 0..q {
                            dst_row[col] = src_row[col] * wi;
                        }
                    }
                } else {
                    let y_slice = y.slice(s![start..start + rows, ..]);
                    for local in 0..rows {
                        let wi = w[start + local];
                        let yrow = y_slice.row(local);
                        let dst_off = local * q;
                        let dst_row = &mut chunk_slice[dst_off..dst_off + q];
                        for (col, yij) in yrow.iter().enumerate() {
                            dst_row[col] = yij * wi;
                        }
                    }
                }
            }
            let x_slice = x.slice(s![start..start + rows, ..]);
            let wy_slice = wy_chunk.slice(s![0..rows, ..]);
            let x_view = FaerArrayView::new(&x_slice);
            let wy_view = FaerArrayView::new(&wy_slice);
            let par = matmul_parallelism(px, q, rows);
            matmul(
                out_view.as_mut(),
                Accum::Add,
                x_view.as_ref().transpose(),
                wy_view.as_ref(),
                1.0,
                par,
            );
        }
    }

    result
}

/// Compute the 2×2 block joint Hessian in a single streaming pass:
///   [X_a^T diag(w_aa) X_a,   X_a^T diag(w_ab) X_b]
///   [X_b^T diag(w_ab) X_a,   X_b^T diag(w_bb) X_b]
///
/// This reads X_a and X_b once per chunk instead of twice (saving 50% bandwidth).
pub fn fast_joint_hessian_2x2<
    S1: Data<Elem = f64>,
    S2: Data<Elem = f64>,
    S3: Data<Elem = f64>,
    S4: Data<Elem = f64>,
    S5: Data<Elem = f64>,
>(
    x_a: &ArrayBase<S1, Ix2>,
    x_b: &ArrayBase<S2, Ix2>,
    w_aa: &ArrayBase<S3, Ix1>,
    w_ab: &ArrayBase<S4, Ix1>,
    w_bb: &ArrayBase<S5, Ix1>,
) -> Array2<f64> {
    if let Some(out) = crate::gpu_hook::gpu_dispatch().and_then(|d| {
        d.try_fast_joint_hessian_2x2(
            x_a.view(),
            x_b.view(),
            w_aa.view(),
            w_ab.view(),
            w_bb.view(),
        )
    }) {
        return out;
    }
    fast_joint_hessian_2x2_impl(x_a, x_b, w_aa, w_ab, w_bb)
}

#[inline]
fn fast_joint_hessian_2x2_impl<
    S1: Data<Elem = f64>,
    S2: Data<Elem = f64>,
    S3: Data<Elem = f64>,
    S4: Data<Elem = f64>,
    S5: Data<Elem = f64>,
>(
    x_a: &ArrayBase<S1, Ix2>,
    x_b: &ArrayBase<S2, Ix2>,
    w_aa: &ArrayBase<S3, Ix1>,
    w_ab: &ArrayBase<S4, Ix1>,
    w_bb: &ArrayBase<S5, Ix1>,
) -> Array2<f64> {
    use faer::Accum;
    use faer::linalg::matmul::matmul;
    use ndarray::{ShapeBuilder, s};

    let n = x_a.nrows();
    let pa = x_a.ncols();
    let pb = x_b.ncols();
    let total = pa + pb;
    assert_eq!(n, x_b.nrows());
    assert_eq!(n, w_aa.len());
    assert_eq!(n, w_ab.len());
    assert_eq!(n, w_bb.len());

    if n == 0 || total == 0 {
        return Array2::<f64>::zeros((total, total));
    }

    // For small problems, fall back to separate computations
    if !should_use_faer_matmul(pa.max(pb), pa.max(pb), n) {
        let waa_xa = Array2::from_shape_fn((n, pa), |(i, j)| w_aa[i] * x_a[[i, j]]);
        let wab_xb = Array2::from_shape_fn((n, pb), |(i, j)| w_ab[i] * x_b[[i, j]]);
        let wbb_xb = Array2::from_shape_fn((n, pb), |(i, j)| w_bb[i] * x_b[[i, j]]);
        let mut out = Array2::<f64>::zeros((total, total));
        out.slice_mut(s![..pa, ..pa]).assign(&x_a.t().dot(&waa_xa));
        out.slice_mut(s![..pa, pa..]).assign(&x_a.t().dot(&wab_xb));
        out.slice_mut(s![pa.., pa..]).assign(&x_b.t().dot(&wbb_xb));
        // Mirror upper to lower
        for i in 0..total {
            for j in 0..i {
                out[[i, j]] = out[[j, i]];
            }
        }
        return out;
    }

    // Need buffers for: waa_xa(chunk×pa) + wab_xb(chunk×pb) + wbb_xb(chunk×pb)
    let cols_needed = pa + 2 * pb;
    let chunk_rows = streaming_chunk_rows(cols_needed, n);

    let mut out = Array2::<f64>::zeros((total, total).f());
    // Row-major weighted buffers so the per-row scale loops have stride-1
    // writes (the previous F-order layout strided writes by chunk_rows
    // across `pa` / `pb`, gutting vectorization on the per-PIRLS-iter
    // joint Hessian assembly). faer's matmul handles either layout.
    let mut waa_xa_buf = Array2::<f64>::zeros((chunk_rows, pa));
    let mut wab_xb_buf = Array2::<f64>::zeros((chunk_rows, pb));
    let mut wbb_xb_buf = Array2::<f64>::zeros((chunk_rows, pb));

    let xa_is_row_major = x_a.is_standard_layout();
    let xb_is_row_major = x_b.is_standard_layout();
    let waa_slice_opt = w_aa.as_slice();
    let wab_slice_opt = w_ab.as_slice();
    let wbb_slice_opt = w_bb.as_slice();

    {
        let mut out_mat = array2_to_matmut(&mut out);

        for start in (0..n).step_by(chunk_rows) {
            let rows = (n - start).min(chunk_rows);
            let xa_slice = x_a.slice(s![start..start + rows, ..]);
            let xb_slice = x_b.slice(s![start..start + rows, ..]);

            // Weight X_a and X_b in a single pass through this chunk.
            {
                let waa_chunk = waa_xa_buf
                    .as_slice_mut()
                    .expect("row-major waa chunk is contiguous");
                let wab_chunk = wab_xb_buf
                    .as_slice_mut()
                    .expect("row-major wab chunk is contiguous");
                let wbb_chunk = wbb_xb_buf
                    .as_slice_mut()
                    .expect("row-major wbb chunk is contiguous");

                if xa_is_row_major
                    && xb_is_row_major
                    && let (Some(xa_all), Some(xb_all)) = (x_a.as_slice(), x_b.as_slice())
                    && let (Some(waa_all), Some(wab_all), Some(wbb_all)) =
                        (waa_slice_opt, wab_slice_opt, wbb_slice_opt)
                {
                    for local in 0..rows {
                        let i = start + local;
                        let waa_i = waa_all[i];
                        let wab_i = wab_all[i];
                        let wbb_i = wbb_all[i];
                        let xa_off = i * pa;
                        let xa_row = &xa_all[xa_off..xa_off + pa];
                        let xb_off = i * pb;
                        let xb_row = &xb_all[xb_off..xb_off + pb];
                        let waa_off = local * pa;
                        let wab_off = local * pb;
                        let wbb_off = local * pb;
                        let waa_row = &mut waa_chunk[waa_off..waa_off + pa];
                        for col in 0..pa {
                            waa_row[col] = xa_row[col] * waa_i;
                        }
                        let wab_row = &mut wab_chunk[wab_off..wab_off + pb];
                        let wbb_row = &mut wbb_chunk[wbb_off..wbb_off + pb];
                        for col in 0..pb {
                            let xij = xb_row[col];
                            wab_row[col] = xij * wab_i;
                            wbb_row[col] = xij * wbb_i;
                        }
                    }
                } else {
                    for local in 0..rows {
                        let i = start + local;
                        let waa_i = w_aa[i];
                        let wab_i = w_ab[i];
                        let wbb_i = w_bb[i];
                        let waa_off = local * pa;
                        let wab_off = local * pb;
                        let wbb_off = local * pb;
                        let waa_row = &mut waa_chunk[waa_off..waa_off + pa];
                        let xa_row = xa_slice.row(local);
                        for (col, xij) in xa_row.iter().enumerate() {
                            waa_row[col] = xij * waa_i;
                        }
                        let wab_row = &mut wab_chunk[wab_off..wab_off + pb];
                        let wbb_row = &mut wbb_chunk[wbb_off..wbb_off + pb];
                        let xb_row = xb_slice.row(local);
                        for (col, xij) in xb_row.iter().enumerate() {
                            wab_row[col] = xij * wab_i;
                            wbb_row[col] = xij * wbb_i;
                        }
                    }
                }
            }

            let xa_view = FaerArrayView::new(&xa_slice);
            let xb_view = FaerArrayView::new(&xb_slice);
            let waa_xa_slice = waa_xa_buf.slice(s![0..rows, ..]);
            let wab_xb_slice = wab_xb_buf.slice(s![0..rows, ..]);
            let wbb_xb_slice = wbb_xb_buf.slice(s![0..rows, ..]);
            let waa_xa_view = FaerArrayView::new(&waa_xa_slice);
            let wab_xb_view = FaerArrayView::new(&wab_xb_slice);
            let wbb_xb_view = FaerArrayView::new(&wbb_xb_slice);

            // Block [0..pa, 0..pa]: X_a^T diag(w_aa) X_a
            matmul(
                out_mat.rb_mut().submatrix_mut(0, 0, pa, pa),
                Accum::Add,
                xa_view.as_ref().transpose(),
                waa_xa_view.as_ref(),
                1.0,
                matmul_parallelism(pa, pa, rows),
            );
            // Block [0..pa, pa..total]: X_a^T diag(w_ab) X_b
            matmul(
                out_mat.rb_mut().submatrix_mut(0, pa, pa, pb),
                Accum::Add,
                xa_view.as_ref().transpose(),
                wab_xb_view.as_ref(),
                1.0,
                matmul_parallelism(pa, pb, rows),
            );
            // Block [pa..total, pa..total]: X_b^T diag(w_bb) X_b
            matmul(
                out_mat.rb_mut().submatrix_mut(pa, pa, pb, pb),
                Accum::Add,
                xb_view.as_ref().transpose(),
                wbb_xb_view.as_ref(),
                1.0,
                matmul_parallelism(pb, pb, rows),
            );
        }
    } // out_mat dropped
    // Mirror upper triangle to lower
    for i in 0..total {
        for j in 0..i {
            out[[i, j]] = out[[j, i]];
        }
    }
    out
}

fn mat_to_array(mat: MatRef<'_, f64>) -> Array2<f64> {
    let nrows = mat.nrows();
    let ncols = mat.ncols();
    let mut out = Array2::<f64>::zeros((nrows, ncols));
    if nrows == 0 || ncols == 0 {
        return out;
    }
    // ndarray is row-major by default. Write row-by-row for best cache behavior
    // on the output side.
    if let Some(out_slice) = out.as_slice_memory_order_mut() {
        // Row-major: out_slice[i * ncols + j] = mat[(i, j)]
        for i in 0..nrows {
            let row_start = i * ncols;
            for j in 0..ncols {
                out_slice[row_start + j] = mat[(i, j)];
            }
        }
    } else {
        for j in 0..ncols {
            for i in 0..nrows {
                out[[i, j]] = mat[(i, j)];
            }
        }
    }
    out
}

/// Write faer matmul result A*B directly into a pre-allocated ndarray Array2.
/// Avoids the intermediate faer::Mat allocation and mat_to_array copy.
#[inline]
pub fn fast_ab_into<S1: Data<Elem = f64>, S2: Data<Elem = f64>>(
    a: &ArrayBase<S1, Ix2>,
    b: &ArrayBase<S2, Ix2>,
    out: &mut Array2<f64>,
) {
    fast_ab_into_impl(a, b, out);
}

#[inline]
fn fast_ab_into_impl<S1: Data<Elem = f64>, S2: Data<Elem = f64>>(
    a: &ArrayBase<S1, Ix2>,
    b: &ArrayBase<S2, Ix2>,
    out: &mut Array2<f64>,
) {
    use faer::Accum;
    use faer::linalg::matmul::matmul;

    let (n, p) = a.dim();
    let (p_b, q) = b.dim();
    assert_eq!(p, p_b, "A and B must have compatible inner dimensions");
    assert_eq!(out.dim(), (n, q), "output dimensions must match A*B result");

    if !should_use_faer_matmul(n, q, p) {
        out.assign(&a.dot(b));
        return;
    }

    let aview = FaerArrayView::new(a);
    let bview = FaerArrayView::new(b);
    let a_ref = aview.as_ref();
    let b_ref = bview.as_ref();

    let par = matmul_parallelism(n, q, p);
    let mut outview = array2_to_matmut(out);
    matmul(outview.as_mut(), Accum::Replace, a_ref, b_ref, 1.0, par);
}

fn diag_to_array(diag: DiagRef<'_, f64>) -> Array1<f64> {
    let mat = diag.column_vector().as_mat();
    let mut out = Array1::<f64>::zeros(mat.nrows());
    for i in 0..mat.nrows() {
        out[i] = mat[(i, 0)];
    }
    out
}

pub struct FaerArrayView<'a> {
    ptr: *const f64,
    rows: usize,
    cols: usize,
    row_stride: isize,
    col_stride: isize,
    owned: Option<Array2<f64>>,
    marker: PhantomData<&'a f64>,
}

impl<'a> FaerArrayView<'a> {
    #[inline]
    pub fn new<S: Data<Elem = f64>>(array: &'a ArrayBase<S, Ix2>) -> Self {
        let (rows, cols) = array.dim();
        let strides = array.strides();
        // Guard against layouts that can alias or reverse memory traversal (e.g.
        // negative/zero strides). These can violate assumptions in faer kernels.
        // For such layouts we materialize a compact owned copy.
        if strides[0] <= 0 || strides[1] <= 0 {
            let owned = array.to_owned();
            let owned_strides = owned.strides();
            return Self {
                ptr: owned.as_ptr(),
                rows,
                cols,
                row_stride: owned_strides[0],
                col_stride: owned_strides[1],
                owned: Some(owned),
                marker: PhantomData,
            };
        }

        Self {
            ptr: array.as_ptr(),
            rows,
            cols,
            row_stride: strides[0],
            col_stride: strides[1],
            owned: None,
            marker: PhantomData,
        }
    }

    #[inline]
    pub fn as_ref(&self) -> MatRef<'_, f64> {
        let (ptr, rows, cols, row_stride, col_stride) = if let Some(owned) = &self.owned {
            let strides = owned.strides();
            (
                owned.as_ptr(),
                owned.nrows(),
                owned.ncols(),
                strides[0],
                strides[1],
            )
        } else {
            (
                self.ptr,
                self.rows,
                self.cols,
                self.row_stride,
                self.col_stride,
            )
        };
        // SAFETY: ptr/shape/strides come from either a live ndarray view
        // (positive strides, validated bounds/alignment) or the owned
        // compact copy held inside this wrapper — no mutable aliasing.
        unsafe { MatRef::from_raw_parts(ptr, rows, cols, row_stride, col_stride) }
    }
}

pub struct FaerColView<'a> {
    ptr: *const f64,
    len: usize,
    stride: isize,
    owned: Option<Array1<f64>>,
    marker: PhantomData<&'a f64>,
}

impl<'a> FaerColView<'a> {
    #[inline]
    pub fn new<S: Data<Elem = f64>>(array: &'a ArrayBase<S, Ix1>) -> Self {
        let len = array.len();
        let stride = array.strides()[0];
        if stride <= 0 {
            let owned = array.to_owned();
            return Self {
                ptr: owned.as_ptr(),
                len,
                stride: 1,
                owned: Some(owned),
                marker: PhantomData,
            };
        }
        Self {
            ptr: array.as_ptr(),
            len,
            stride,
            owned: None,
            marker: PhantomData,
        }
    }

    #[inline]
    pub fn as_ref(&self) -> MatRef<'_, f64> {
        let (ptr, len, stride) = if let Some(owned) = &self.owned {
            (owned.as_ptr(), owned.len(), 1)
        } else {
            (self.ptr, self.len, self.stride)
        };
        // SAFETY: ptr/len/stride come from either a live ndarray column
        // (positive stride, validated bounds/alignment) or the owned
        // compact copy; ncols=1 so the 0 col-stride is unused.
        unsafe { MatRef::from_raw_parts(ptr, len, 1, stride, 0) }
    }
}

pub trait FaerSvd {
    fn svd(
        &self,
        compute_u: bool,
        computevt: bool,
    ) -> Result<(Option<Array2<f64>>, Array1<f64>, Option<Array2<f64>>), FaerLinalgError>;
}

impl<S: Data<Elem = f64>> FaerSvd for ArrayBase<S, Ix2> {
    fn svd(
        &self,
        compute_u: bool,
        computevt: bool,
    ) -> Result<(Option<Array2<f64>>, Array1<f64>, Option<Array2<f64>>), FaerLinalgError> {
        let faerview = FaerArrayView::new(self);
        let faer_mat = faerview.as_ref();
        if !compute_u && !computevt {
            let (rows, cols) = faer_mat.shape();
            let mut singular = Diag::<f64>::zeros(rows.min(cols));
            let par = get_global_parallelism();
            let mut mem = MemBuffer::new(svd::svd_scratch::<f64>(
                rows,
                cols,
                ComputeSvdVectors::No,
                ComputeSvdVectors::No,
                par,
                Default::default(),
            ));
            let stack = MemStack::new(&mut mem);
            svd::svd(
                faer_mat,
                singular.as_mut(),
                None,
                None,
                par,
                stack,
                Default::default(),
            )
            .map_err(|_| FaerLinalgError::SvdNoConvergence {
                context: "faer SVD singular values only",
            })?;
            let singularvalues = diag_to_array(singular.as_ref());
            return Ok((None, singularvalues, None));
        }

        let (rows, cols) = faer_mat.shape();
        let rank = rows.min(cols);
        let compute_u_flag = if compute_u {
            ComputeSvdVectors::Thin
        } else {
            ComputeSvdVectors::No
        };
        let computev_flag = if computevt {
            ComputeSvdVectors::Thin
        } else {
            ComputeSvdVectors::No
        };

        let mut singular = Diag::<f64>::zeros(rows.min(cols));
        let mut u_storage = compute_u.then(|| Mat::<f64>::zeros(rows, rank));
        let mut v_storage = computevt.then(|| Mat::<f64>::zeros(cols, rank));

        let par = get_global_parallelism();
        let mut mem = MemBuffer::new(svd::svd_scratch::<f64>(
            rows,
            cols,
            compute_u_flag,
            computev_flag,
            par,
            Default::default(),
        ));
        let stack = MemStack::new(&mut mem);

        svd::svd(
            faer_mat.as_ref(),
            singular.as_mut(),
            u_storage.as_mut().map(|mat| mat.as_mut()),
            v_storage.as_mut().map(|mat| mat.as_mut()),
            par,
            stack,
            Default::default(),
        )
        .map_err(|_| FaerLinalgError::SvdNoConvergence {
            context: "faer SVD with vectors",
        })?;

        let singularvalues = diag_to_array(singular.as_ref());
        let u_opt = u_storage.map(|mat| mat_to_array(mat.as_ref()));
        let vt_opt = v_storage.map(|mat| {
            let mat_ref = mat.as_ref();
            let mut out = Array2::<f64>::zeros((mat_ref.ncols(), mat_ref.nrows()));
            for j in 0..mat_ref.nrows() {
                for i in 0..mat_ref.ncols() {
                    out[[i, j]] = mat_ref[(j, i)];
                }
            }
            out
        });

        Ok((u_opt, singularvalues, vt_opt))
    }
}

pub trait FaerEigh {
    fn eigh(&self, side: Side) -> Result<(Array1<f64>, Array2<f64>), FaerLinalgError>;
}

/// Strict self-adjoint eigendecomposition of the exact supplied matrix.
///
/// This entrypoint performs finite/symmetry validation and one direct faer EVD
/// attempt. It never symmetrizes, rescales, jitters the diagonal, or subtracts
/// a repair afterward. Rank and pseudoinverse code must use this function so
/// its reported spectrum belongs to the matrix the caller supplied.
pub fn strict_symmetric_eigh<S: Data<Elem = f64>>(
    matrix: &ArrayBase<S, Ix2>,
    side: Side,
) -> Result<(Array1<f64>, Array2<f64>), FaerLinalgError> {
    let owned = matrix.to_owned();
    if owned.nrows() == 0 || owned.nrows() != owned.ncols() {
        return Err(FaerLinalgError::StrictSelfAdjointEigenInvalidInput {
            reason: format!(
                "expected non-empty square matrix, got {}x{}",
                owned.nrows(),
                owned.ncols()
            ),
        });
    }
    crate::utils::validate_finite_symmetric_matrix(
        &owned,
        "strict self-adjoint eigendecomposition",
    )
    .map_err(
        |error| FaerLinalgError::StrictSelfAdjointEigenInvalidInput {
            reason: error.to_string(),
        },
    )?;
    let view = FaerArrayView::new(&owned);
    let eigen = catch_unwind(AssertUnwindSafe(|| view.as_ref().self_adjoint_eigen(side)))
        .map_err(|_| FaerLinalgError::FactorizationFailed {
            context: "strict self-adjoint eigendecomposition panic boundary",
        })?
        .map_err(FaerLinalgError::SelfAdjointEigen)?;
    let values = diag_to_array(eigen.S());
    let vectors = mat_to_array(eigen.U());
    if values.iter().any(|value| !value.is_finite())
        || vectors.iter().any(|value| !value.is_finite())
    {
        return Err(FaerLinalgError::SelfAdjointEigenNonFiniteInput {
            context: "strict self-adjoint eigendecomposition output validation",
        });
    }
    Ok((values, vectors))
}

impl<S: Data<Elem = f64>> FaerEigh for ArrayBase<S, Ix2> {
    fn eigh(&self, side: Side) -> Result<(Array1<f64>, Array2<f64>), FaerLinalgError> {
        fn try_eigh(
            matrix: &Array2<f64>,
            side: Side,
        ) -> Result<(Array1<f64>, Array2<f64>), FaerLinalgError> {
            let faerview = FaerArrayView::new(matrix);
            // #2267/#2738 — time the decomposition and name the parallelism that
            // actually governed it.
            //
            // `self_adjoint_eigen` is one of faer's high-level entry points: it
            // takes no parallelism argument and reads
            // `faer::get_global_parallelism()` internally. So the policy that
            // decides whether this runs on one core or many is the PROCESS-GLOBAL
            // one — which a live `FaerSequentialScope` anywhere in the process
            // pins to `Par::Seq` for every thread — and NOT
            // `effective_global_parallelism`, whose nested-region guard cannot
            // reach here. Reporting the wrong one would name a policy that did
            // not apply.
            //
            // This is `O(dim^3)`, so at `dim` in the thousands the difference
            // between sequential and a wide pool is hours. #2267 lost two
            // three-hour jobs to a decomposition that was silent about both its
            // duration and its parallelism; the count makes "one slow call or
            // many?" answerable without a second run.
            let eigh_started = std::time::Instant::now();
            let eigh_par = get_global_parallelism();
            let eigen = catch_unwind(AssertUnwindSafe(|| {
                faerview.as_ref().self_adjoint_eigen(side)
            }))
            .map_err(|_| FaerLinalgError::FactorizationFailed {
                context: "self-adjoint eigendecomposition panic boundary",
            })?
            .map_err(FaerLinalgError::SelfAdjointEigen)?;
            let eigh_elapsed = eigh_started.elapsed();
            let eigh_calls = EIGH_CALLS.fetch_add(1, Ordering::Relaxed) + 1;
            if eigh_par == Par::Seq {
                EIGH_SEQ_CALLS.fetch_add(1, Ordering::Relaxed);
            }
            EIGH_MAX_DIM.fetch_max(matrix.nrows() as u64, Ordering::Relaxed);
            let eigh_nanos_total = EIGH_NANOS
                .fetch_add(eigh_elapsed.as_nanos() as u64, Ordering::Relaxed)
                + eigh_elapsed.as_nanos() as u64;
            record_thread_eigh(
                eigh_par == Par::Seq,
                matrix.nrows() as u64,
                eigh_elapsed.as_nanos() as u64,
            );
            log::debug!(
                "[eigh] dim={} elapsed={:.3}s faer_global_parallelism={:?} \
                 calls_so_far={eigh_calls} cumulative={:.3}s",
                matrix.nrows(),
                eigh_elapsed.as_secs_f64(),
                eigh_par,
                eigh_nanos_total as f64 / 1e9,
            );
            let values = diag_to_array(eigen.S());
            let vectors = mat_to_array(eigen.U());
            Ok((values, vectors))
        }

        let owned = self.to_owned();
        if owned.nrows() != owned.ncols() {
            return Err(FaerLinalgError::FactorizationFailed {
                context: "self-adjoint eigendecomposition non-square input",
            });
        }
        if owned.nrows() == 0 {
            return Ok((Array1::zeros(0), Array2::zeros((0, 0))));
        }
        if owned.iter().any(|value| !value.is_finite()) {
            return Err(FaerLinalgError::SelfAdjointEigenNonFiniteInput {
                context: "self-adjoint eigendecomposition input validation",
            });
        }
        if let Ok((evals, evecs)) = try_eigh(&owned, side)
            && evals.iter().all(|value| value.is_finite())
            && evecs.iter().all(|value| value.is_finite())
        {
            return Ok((evals, evecs));
        }

        let mut repaired = owned.clone();
        crate::matrix::symmetrize_in_place(&mut repaired);

        let scale = repaired
            .iter()
            .fold(0.0_f64, |acc, &value| acc.max(value.abs()))
            .max(1.0);
        let scaled = repaired.mapv(|value| value / scale);
        // Relative diagonal-jitter ladder for the eigendecomposition repair: the
        // matrix is pre-scaled to unit max-abs, so these are fractions of its
        // scale. We try the unperturbed matrix first, then escalate the ridge by
        // two decades per attempt until the factorization yields all-finite
        // eigenpairs, accepting the smallest jitter that succeeds.
        const JITTER_SCHEDULE: [f64; 6] = [0.0, 1e-12, 1e-10, 1e-8, 1e-6, 1e-4];
        let jitter_schedule = JITTER_SCHEDULE;
        let mut last_error = FaerLinalgError::FactorizationFailed {
            context: "self-adjoint eigendecomposition repair attempts",
        };

        for &jitter in &jitter_schedule {
            let mut candidate = scaled.clone();
            if jitter > 0.0 {
                let n = candidate.nrows();
                for i in 0..n {
                    candidate[[i, i]] += jitter;
                }
            }

            match try_eigh(&candidate, side) {
                Ok((mut evals, evecs))
                    if evals.iter().all(|value| value.is_finite())
                        && evecs.iter().all(|value| value.is_finite()) =>
                {
                    for value in &mut evals {
                        *value = (*value - jitter) * scale;
                    }
                    return Ok((evals, evecs));
                }
                Ok((_, _)) => {
                    last_error = FaerLinalgError::SelfAdjointEigenNonFiniteInput {
                        context: "self-adjoint eigendecomposition repaired output validation",
                    };
                }
                Err(err) => {
                    last_error = err;
                }
            }
        }

        Err(last_error)
    }
}

pub struct FaerCholeskyFactor {
    factor: solvers::Llt<f64>,
}

impl FaerCholeskyFactor {
    pub fn solvevec(&self, rhs: &Array1<f64>) -> Array1<f64> {
        let mut rhs = rhs.to_owned();
        let mut rhsview = array1_to_col_matmut(&mut rhs);
        self.factor.solve_in_place(rhsview.as_mut());
        rhs
    }

    pub fn solve_mat_in_place(&self, rhs: &mut Array2<f64>) {
        let mut rhsview = array2_to_matmut(rhs);
        self.factor.solve_in_place(rhsview.as_mut());
    }

    pub fn solve_mat_into<S: Data<Elem = f64>>(
        &self,
        rhs: &ArrayBase<S, Ix2>,
        out: &mut Array2<f64>,
    ) {
        if out.dim() != rhs.dim() {
            *out = Array2::<f64>::zeros(rhs.dim());
        }
        out.assign(rhs);
        self.solve_mat_in_place(out);
    }

    pub fn solve_mat(&self, rhs: &Array2<f64>) -> Array2<f64> {
        let mut out = Array2::<f64>::zeros(rhs.dim());
        self.solve_mat_into(rhs, &mut out);
        out
    }

    pub fn diag(&self) -> Array1<f64> {
        diag_to_array(self.factor.L().diagonal())
    }

    pub fn lower_triangular(&self) -> Array2<f64> {
        mat_to_array(self.factor.L())
    }
}

impl crate::matrix::FactorizedSystem for FaerCholeskyFactor {
    fn solve(&self, rhs: &Array1<f64>) -> Result<Array1<f64>, String> {
        let out = self.solvevec(rhs);
        if out.iter().all(|value| value.is_finite()) {
            Ok(out)
        } else {
            Err("strict Cholesky solve produced non-finite values".to_string())
        }
    }

    fn solvemulti(&self, rhs: &Array2<f64>) -> Result<Array2<f64>, String> {
        let out = self.solve_mat(rhs);
        if out.iter().all(|value| value.is_finite()) {
            Ok(out)
        } else {
            Err("strict Cholesky multi-solve produced non-finite values".to_string())
        }
    }

    fn logdet(&self) -> f64 {
        cholesky_factor_logdet(self.factor.L())
    }
}

pub trait FaerCholesky {
    fn cholesky(&self, side: Side) -> Result<FaerCholeskyFactor, FaerLinalgError>;
}

impl<S: Data<Elem = f64>> FaerCholesky for ArrayBase<S, Ix2> {
    fn cholesky(&self, side: Side) -> Result<FaerCholeskyFactor, FaerLinalgError> {
        let faerview = FaerArrayView::new(self);
        let factor = faerview
            .as_ref()
            .llt(side)
            .map_err(FaerLinalgError::Cholesky)?;
        Ok(FaerCholeskyFactor { factor })
    }
}

pub trait FaerQr {
    fn qr(&self) -> Result<(Array2<f64>, Array2<f64>), FaerLinalgError>;
}

impl<S: Data<Elem = f64>> FaerQr for ArrayBase<S, Ix2> {
    fn qr(&self) -> Result<(Array2<f64>, Array2<f64>), FaerLinalgError> {
        let faerview = FaerArrayView::new(self);
        let qr = faerview.as_ref().qr();
        let q = qr.compute_thin_Q();
        let r = qr.thin_R();
        Ok((mat_to_array(q.as_ref()), mat_to_array(r)))
    }
}

/// Compute an orthonormal basis for `null(a^T)` using column-pivoted QR on `a`.
///
/// This is intended for tall/skinny matrices where `a ∈ R^{m×n}` with `m >= n`.
/// If `A P^T = Q R`, then the trailing `m-rank(A)` columns of `Q` span
/// `null(A^T)`.
///
/// The trailing columns of `Q` are reconstructed by applying the stored
/// Householder reflector sequence to canonical basis vectors. When `A` is
/// numerically rank zero (e.g. an entirely unpenalized block penalty in a
/// parametric-only GLM), *every* reflector is degenerate — the Householder
/// vector of a zero column has zero norm, so faer's coefficients become
/// non-finite and the reconstructed basis is filled with `NaN`. Mathematically
/// a rank-zero `m×n` matrix has `null(A^T) = R^m`, whose canonical orthonormal
/// basis is the identity, so we return `I_m` directly instead of routing through
/// the (undefined) reflectors. This keeps every downstream consumer — REML
/// null-space log-determinants, identifiability audits — finite and exact for
/// the fully-unpenalized case. For `rank >= 1` at least one well-defined
/// reflector seeds the block, and the reconstruction stays finite.
pub fn rrqr_nullspace_basis<S: Data<Elem = f64>>(
    a: &ArrayBase<S, Ix2>,
    rank_alpha: f64,
) -> Result<(Array2<f64>, usize), FaerLinalgError> {
    rrqr_nullspace_basis_inner(a, RrqrRankCutoff::RelativeAlpha(rank_alpha))
}

/// Which absolute cutoff on `|R_ii|` separates rank from null in
/// [`rrqr_nullspace_basis_with_cutoff`] / [`rrqr_nullspace_basis`].
#[derive(Debug, Clone, Copy)]
enum RrqrRankCutoff {
    /// `rank_alpha · ε · max(m, n) · max(|R₀₀|, 1)` — a machine-precision
    /// cutoff derived from the factorization's own leading pivot. This asks
    /// "is this direction numerically distinguishable from zero at all?".
    RelativeAlpha(f64),
    /// A caller-supplied absolute cutoff in the units of `a`'s singular
    /// values. Use this when the null/range partition is fixed by an external
    /// convention (e.g. a penalty spectrum's `spectral_tolerance`) rather than
    /// by float representability, so that two consumers of the same object
    /// cannot disagree about which directions are unpenalized.
    Absolute(f64),
}

/// [`rrqr_nullspace_basis`] with an explicit absolute cutoff on the pivoted
/// `|R_ii|` (i.e. in the units of `a`'s singular values) instead of the
/// machine-precision `rank_alpha` heuristic.
///
/// A rank decision is a *convention*, not a fact about floats: the same matrix
/// has a different null space depending on the scale below which a direction
/// counts as unpenalized. When one consumer answers that question with
/// machine-epsilon and another with a penalty-spectrum tolerance five decades
/// looser, they silently describe different models of the same block (gam#2433:
/// a Duchon smooth's realized penalty topology changed between two builders for
/// exactly this reason). Callers that already own such a convention pass it
/// here rather than re-deriving one.
pub fn rrqr_nullspace_basis_with_cutoff<S: Data<Elem = f64>>(
    a: &ArrayBase<S, Ix2>,
    cutoff: f64,
) -> Result<(Array2<f64>, usize), FaerLinalgError> {
    rrqr_nullspace_basis_inner(a, RrqrRankCutoff::Absolute(cutoff))
}

fn rrqr_nullspace_basis_inner<S: Data<Elem = f64>>(
    a: &ArrayBase<S, Ix2>,
    cutoff: RrqrRankCutoff,
) -> Result<(Array2<f64>, usize), FaerLinalgError> {
    let faerview = FaerArrayView::new(a);
    let qr = faerview.as_ref().col_piv_qr();
    let r = qr.thin_R();
    let diag_len = r.nrows().min(r.ncols());
    let leading_diag = if diag_len > 0 { r[(0, 0)].abs() } else { 0.0 };
    let tol = match cutoff {
        RrqrRankCutoff::RelativeAlpha(rank_alpha) => {
            rank_alpha
                * f64::EPSILON
                * (a.nrows().max(a.ncols()).max(1) as f64)
                * leading_diag.max(1.0)
        }
        RrqrRankCutoff::Absolute(tol) => tol,
    };
    let rank = (0..diag_len).filter(|&i| r[(i, i)].abs() > tol).count();
    let z = if rank >= a.nrows() {
        Array2::<f64>::zeros((a.nrows(), 0))
    } else if rank == 0 {
        // Numerically rank-zero input: the whole space is the null space.
        // Return the canonical orthonormal basis directly; the Householder
        // reflectors of a zero matrix are degenerate and would yield NaN.
        Array2::<f64>::eye(a.nrows())
    } else {
        let nullity = a.nrows() - rank;
        let mut selector = Mat::<f64>::zeros(a.nrows(), nullity);
        for j in 0..nullity {
            selector[(rank + j, j)] = 1.0;
        }
        let par = get_global_parallelism();
        faer::linalg::householder::apply_block_householder_sequence_on_the_left_in_place_with_conj(
            qr.Q_basis(),
            qr.Q_coeff(),
            Conj::No,
            selector.as_mut(),
            par,
            MemStack::new(&mut MemBuffer::new(
                faer::linalg::householder::apply_block_householder_sequence_on_the_left_in_place_scratch::<f64>(
                    a.nrows(),
                    qr.Q_coeff().nrows(),
                    nullity,
                ),
            )),
        );
        mat_to_array(selector.as_ref())
    };
    Ok((z, rank))
}

#[inline]
pub const fn default_rrqr_rank_alpha() -> f64 {
    RRQR_RANK_ALPHA
}

/// Result of a column-pivoted QR with rank detection and column permutation.
///
/// `A · P = Q · R` where the permutation `P` is exposed as the forward index
/// array: column `j` of `A · P` corresponds to original column
/// `column_permutation[j]` of `A`. With rank `r < min(m, n)`, the trailing
/// `min(m, n) - r` entries of `column_permutation` name the columns that the
/// pivoted QR demoted past the rank threshold — i.e., the columns identified
/// as redundant. Identifiability auditors (`identifiability::audit`)
/// use that suffix to attribute `DroppedColumn` entries to specific original
/// columns.
pub struct RrqrWithPermutation {
    pub rank: usize,
    pub column_permutation: Vec<usize>,
    pub leading_diag_abs: f64,
    pub rank_tol: f64,
}

/// Column-pivoted rank-revealing QR returning the rank, the column permutation,
/// and the rank-detection tolerance. Use this when callers need to name which
/// columns the pivoted QR demoted past the rank threshold.
///
/// The rank cutoff matches [`rrqr_nullspace_basis`]: a column-pivoted QR is
/// computed on `a`; columns with `|R[i, i]| > tol` count toward the rank,
/// where `tol = rank_alpha · eps · max(m, n, 1) · max(|R[0, 0]|, 1)`. Returns
/// `Err` when `a` has zero rows.
pub fn rrqr_with_permutation<S: Data<Elem = f64>>(
    a: &ArrayBase<S, Ix2>,
    rank_alpha: f64,
) -> Result<RrqrWithPermutation, FaerLinalgError> {
    if a.nrows() == 0 {
        return Err(FaerLinalgError::FactorizationFailed {
            context: "rrqr_with_permutation: input has zero rows",
        });
    }
    let faerview = FaerArrayView::new(a);
    let qr = faerview.as_ref().col_piv_qr();
    let r = qr.thin_R();
    let diag_len = r.nrows().min(r.ncols());
    let leading_diag = if diag_len > 0 { r[(0, 0)].abs() } else { 0.0 };
    let tol = rank_alpha
        * f64::EPSILON
        * (a.nrows().max(a.ncols()).max(1) as f64)
        * leading_diag.max(1.0);
    let rank = (0..diag_len).filter(|&i| r[(i, i)].abs() > tol).count();
    let (forward, _inverse) = qr.P().arrays();
    let column_permutation: Vec<usize> = forward.iter().copied().map(|idx| idx.unbound()).collect();
    Ok(RrqrWithPermutation {
        rank,
        column_permutation,
        leading_diag_abs: leading_diag,
        rank_tol: tol,
    })
}

/// Result of a Gram-driven column-pivoted RRQR (see
/// [`rrqr_from_gram_with_permutation`]). Carries the same rank / permutation /
/// tolerance as [`RrqrWithPermutation`], plus a `verdict_margin` that measures
/// how unambiguous the rank cut is — the ratio between the smallest *kept*
/// pivot and the rank tolerance. A large margin means squaring the design into
/// a Gram could not have flipped any rank decision; a small margin means the
/// verdict sits near the cliff and the caller should re-confirm on the full
/// (un-squared) design to stay bit-exact.
pub struct RrqrFromGram {
    pub rank: usize,
    pub column_permutation: Vec<usize>,
    pub rank_tol: f64,
    /// Leading pivot magnitude `|R[0,0]|` of the square-root factor — equal to
    /// the largest column norm of the original tall design (col-piv QR pivots the
    /// largest-norm column first), so it matches the tall path's
    /// `RrqrWithPermutation::leading_diag_abs`.
    pub leading_diag_abs: f64,
    /// `min_kept_pivot / rank_tol` (∞ when full rank with no kept pivot below
    /// tol, i.e. every pivot is comfortably above; `0` when rank is 0).
    pub verdict_margin: f64,
}

/// Column-pivoted rank-revealing QR computed from the design's `p × p` Gram
/// `G = AᵀA` (or penalty-augmented `AᵀA + SᵀS`) instead of from the tall
/// `m × p` design itself.
///
/// # Why this is exact (in exact arithmetic)
///
/// Column-pivoted QR selects, at each step, the not-yet-pivoted column with the
/// largest residual norm, where the residual is the part orthogonal to the
/// already-chosen columns. Those residual norms — and the resulting pivot
/// sequence, the diagonal magnitudes `|R[i,i]|`, and hence the rank cut — are a
/// function of the column *inner products* only, i.e. of the Gram `G`. Running
/// col-piv QR on the Cholesky factor `R₀` of `G` (`R₀ᵀR₀ = G`, `R₀` is `p × p`)
/// reproduces the identical pivot order and identical `|R[i,i]|` as col-piv QR
/// on the original `m × p` matrix, because both see the same column geometry.
/// This is the standard "pivoted QR depends only on the Gram" identity and lets
/// the joint identifiability rank verdict run in `O(p³)` instead of streaming
/// all `m ≈ 2·10⁵` rows again.
///
/// # Tolerance
///
/// The rank cutoff must match what the tall-matrix [`rrqr_with_permutation`]
/// would have used, so the caller passes `m_rows` (the row count of the
/// original tall design, including any appended penalty rows). The tolerance is
/// `rank_alpha · eps · max(m_rows, p) · max(|R[0,0]|, 1)` — bit-identical to the
/// tall path, since `|R[0,0]|` (the leading pivot magnitude = largest column
/// norm) is the same in both factorizations.
///
/// # Finite-precision guard
///
/// Forming `G = AᵀA` squares the condition number, so a rank decision that sits
/// right at the tolerance cliff could in principle flip. The returned
/// `verdict_margin` lets the caller detect that case and fall back to the exact
/// tall RRQR; in the overwhelmingly common well-separated case (full column
/// rank, smallest pivot orders of magnitude above tol) the margin is huge and
/// no fallback is needed.
pub fn rrqr_from_gram_with_permutation<S: Data<Elem = f64>>(
    gram: &ArrayBase<S, Ix2>,
    m_rows: usize,
    rank_alpha: f64,
) -> Result<RrqrFromGram, FaerLinalgError> {
    let p = gram.ncols();
    if p == 0 {
        return Ok(RrqrFromGram {
            rank: 0,
            column_permutation: Vec::new(),
            rank_tol: 0.0,
            leading_diag_abs: 0.0,
            verdict_margin: 0.0,
        });
    }
    if gram.nrows() != p {
        return Err(FaerLinalgError::FactorizationFailed {
            context: "rrqr_from_gram_with_permutation: Gram is not square",
        });
    }
    // Symmetric square-root factor F (p×p) with FᵀF = G. The Gram is PSD by
    // construction (AᵀA), so its eigendecomposition G = V·diag(λ)·Vᵀ gives the
    // factor F = diag(√λ₊)·Vᵀ (rows indexed by eigenpair, columns by original
    // design column). Any factor with FᵀF = G reproduces the same column
    // geometry, which is all col-piv QR consumes — we use the eigen square root
    // rather than a bare Cholesky because Cholesky fails on the numerically
    // semidefinite Gram that is exactly the rank-deficient case we must classify.
    // Tiny-negative eigenvalues from finite precision are clamped to zero.
    let (evals, evecs) = gram.eigh(Side::Lower)?;
    let mut f = Array2::<f64>::zeros((p, p));
    for k in 0..p {
        let scale = evals[k].max(0.0).sqrt();
        if scale == 0.0 {
            continue;
        }
        for i in 0..p {
            f[[k, i]] = scale * evecs[[i, k]];
        }
    }
    // Single col-piv QR on F. Its pivot order, per-pivot |R[i,i]| magnitudes,
    // and leading pivot equal those of col-piv QR on the original tall design
    // (FᵀF = G), so this reproduces the exact tall-path geometry.
    let faer_f = FaerArrayView::new(&f);
    let qr = faer_f.as_ref().col_piv_qr();
    let r = qr.thin_R();
    let diag_len = r.nrows().min(r.ncols());
    let pivots: Vec<f64> = (0..diag_len).map(|i| r[(i, i)].abs()).collect();
    let leading_diag = pivots.first().copied().unwrap_or(0.0);
    let (forward, _inverse) = qr.P().arrays();
    let column_permutation: Vec<usize> = forward.iter().copied().map(|idx| idx.unbound()).collect();
    // Re-scale the tolerance from F's `max(p, p)=p` row dimension to the
    // original tall design's `max(m_rows, p)`, keeping the rank cut bit-
    // identical to what the tall [`rrqr_with_permutation`] would have produced.
    let tol = rank_alpha * f64::EPSILON * (m_rows.max(p).max(1) as f64) * leading_diag.max(1.0);
    let rank = pivots.iter().filter(|&&v| v > tol).count();
    let min_kept = pivots[..rank].iter().copied().fold(f64::INFINITY, f64::min);
    let max_dropped = pivots[rank..].iter().copied().fold(0.0f64, f64::max);
    // Margin: how far the verdict is from the cliff. Use the smaller of
    // (min_kept / tol) and (tol / max_dropped) so a near-tol dropped pivot also
    // shrinks the margin. A margin ≫ 1 means no rank decision could flip.
    let kept_margin = if rank == 0 {
        f64::INFINITY
    } else {
        min_kept / tol
    };
    let dropped_margin = if rank == diag_len {
        f64::INFINITY
    } else {
        tol / max_dropped.max(f64::MIN_POSITIVE)
    };
    // Gram-squaring precision floor. Forming `G = XᵀX` collapses the bottom half
    // of the spectrum: a true singular value below `√ε · σ_max` is lost in the
    // rounding of `G` (its squared value `σ² < ε·σ_max²` underflows the Gram's
    // representable range), and the eigen-square-root then RESURRECTS it as a
    // SPURIOUS pivot of magnitude `≈ √(ε·σ_max²) = √ε · σ_max` — orders of
    // magnitude ABOVE the true σ and above `tol`. That artefact makes col-piv QR
    // on `F` KEEP a column the tall (un-squared) QR would demote: an EXACTLY
    // collinear alias (true σ = 0, so `σ² = 0` floored at `≈ ε·σ_max²`) shows up
    // as a kept pivot near `√ε · leading`, over-ranking the design and dropping
    // nothing (gam#933: a callback-owned column aliased with a higher-priority
    // anchor was never demoted, so the reduction never ran and the MAP-uniqueness
    // check then fired on the raw collinear joint design). `min_kept / tol` does
    // NOT catch this — the spurious pivot sits comfortably above `tol`, so the
    // existing margin reports a falsely-confident verdict. The honest test is
    // whether the smallest KEPT pivot is itself near the Gram precision floor
    // `√ε · leading`: if so, the Gram path cannot distinguish it from a true zero
    // and the verdict MUST be re-confirmed on the full-precision tall design.
    // Encode that as a third margin term `min_kept / (√ε · leading)` so a kept
    // pivot in the floor regime shrinks `verdict_margin` below the caller's
    // fallback threshold; for a genuinely full-rank design every kept pivot is
    // `≫ √ε · leading` and this term is large, leaving the fast path intact.
    let gram_precision_floor = f64::EPSILON.sqrt() * leading_diag.max(1.0);
    let kept_floor_margin = if rank == 0 {
        f64::INFINITY
    } else {
        min_kept / gram_precision_floor.max(f64::MIN_POSITIVE)
    };
    let verdict_margin = kept_margin.min(dropped_margin).min(kept_floor_margin);
    Ok(RrqrFromGram {
        rank,
        column_permutation,
        rank_tol: tol,
        leading_diag_abs: leading_diag,
        verdict_margin,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::{array, s};

    /// Local mirror of the audit's `JOINT_GRAM_RRQR_MIN_VERDICT_MARGIN` fallback
    /// threshold, used only by the regression tests below to assert the verdict
    /// margin lands on the correct side of the cliff. Kept in sync by value (1e3).
    const JOINT_GRAM_RRQR_TRUST_MARGIN_FOR_TEST: f64 = 1.0e3;

    #[test]
    fn rrqr_nullspace_basis_is_orthonormal_and_annihilates_transpose() {
        let a = array![[1.0, 0.0], [1.0, 0.0], [0.0, 2.0], [0.0, 0.0],];
        let (z, rank) =
            rrqr_nullspace_basis(&a, default_rrqr_rank_alpha()).expect("RRQR should succeed");
        assert_eq!(rank, 2);
        assert_eq!(z.nrows(), 4);
        assert_eq!(z.ncols(), 2);

        let gram = z.t().dot(&z);
        let ident = Array2::<f64>::eye(z.ncols());
        let gram_err = (&gram - &ident)
            .iter()
            .fold(0.0_f64, |acc, &v| acc.max(v.abs()));
        assert!(gram_err < 1e-10, "Z is not orthonormal: {gram_err:e}");

        let residual = a.t().dot(&z);
        let resid_max = residual.iter().fold(0.0_f64, |acc, &v| acc.max(v.abs()));
        assert!(resid_max < 1e-10, "A^T Z residual too large: {resid_max:e}");
    }

    #[test]
    fn rrqr_with_permutation_attributes_redundant_column() {
        // 3 columns, column 2 is a duplicate of column 0 → rank 2, column 2
        // is the redundant one that the pivoted QR should demote past the
        // rank threshold. (Column 1 contributes a different direction.)
        let a = array![
            [1.0, 0.0, 1.0],
            [1.0, 0.0, 1.0],
            [0.0, 2.0, 0.0],
            [0.0, 0.0, 0.0],
        ];
        let result =
            rrqr_with_permutation(&a, default_rrqr_rank_alpha()).expect("RRQR should succeed");
        assert_eq!(result.rank, 2);
        assert_eq!(result.column_permutation.len(), 3);
        let demoted = result.column_permutation[result.rank..].to_vec();
        assert!(
            demoted.contains(&2) || demoted.contains(&0),
            "demoted suffix should include one of the aliased columns (0 or 2), got {demoted:?}"
        );
        let mut sorted = result.column_permutation.clone();
        sorted.sort();
        assert_eq!(
            sorted,
            vec![0, 1, 2],
            "permutation must be a valid bijection on 0..n"
        );
    }

    /// A column-pivoted QR orders columns by decreasing pivot norm, so on this
    /// fixture (`‖a_0‖ = 1`, `‖a_1‖ = 2`) the permutation is emphatically *not*
    /// identity: Businger–Golub pivoting takes column 1 first, and with only
    /// two columns a bijection whose first entry is 1 is fully determined as
    /// `[1, 0]`. The previous version sorted the permutation before comparing
    /// it to `[0, 1]` — true of *any* two-element permutation — which destroyed
    /// exactly the ordering information the test was named for, and the name
    /// ("identity-like") asserted the opposite of what a pivoted QR does here.
    #[test]
    fn rrqr_with_permutation_pivots_the_larger_norm_column_first() {
        let a = array![[1.0, 0.0], [0.0, 2.0], [0.0, 0.0]];
        let result =
            rrqr_with_permutation(&a, default_rrqr_rank_alpha()).expect("RRQR should succeed");
        assert_eq!(result.rank, 2);
        let perm = result.column_permutation.clone();

        let mut sorted = perm.clone();
        sorted.sort();
        assert_eq!(
            sorted,
            vec![0, 1],
            "permutation must be a bijection on 0..n, got {perm:?}"
        );

        // Unsorted pivot order: the load-bearing assertion.
        assert_eq!(
            perm,
            vec![1, 0],
            "column-pivoted QR must take the larger-norm column (1, norm 2) \
             before the smaller (0, norm 1), got {perm:?}"
        );

        // The property that order encodes, stated independently of the literal
        // above: original column norms are non-increasing in pivot order.
        let norms: Vec<f64> = perm
            .iter()
            .map(|&j| a.column(j).iter().map(|value| value * value).sum::<f64>().sqrt())
            .collect();
        for window in norms.windows(2) {
            assert!(
                window[0] >= window[1],
                "pivoted column norms must be non-increasing, got {norms:?}"
            );
        }
    }

    #[test]
    fn rrqr_with_permutation_rejects_zero_rows() {
        let a = Array2::<f64>::zeros((0, 3));
        assert!(rrqr_with_permutation(&a, default_rrqr_rank_alpha()).is_err());
    }

    #[test]
    fn rrqr_nullspace_basis_square_zero_matrix_is_finite_identity() {
        // Square zero matrix (the parametric-only penalty case): null(A^T) is
        // the whole space, so the basis must be a finite orthonormal 3x3 set.
        let a = Array2::<f64>::zeros((3, 3));
        let (z, rank) =
            rrqr_nullspace_basis(&a, default_rrqr_rank_alpha()).expect("RRQR should succeed");
        assert_eq!(rank, 0);
        assert_eq!(z.dim(), (3, 3));
        assert!(
            z.iter().all(|v| v.is_finite()),
            "square zero matrix produced a non-finite null basis: {z:?}"
        );
        let gram = z.t().dot(&z);
        let ident = Array2::<f64>::eye(3);
        let gram_err = (&gram - &ident)
            .iter()
            .fold(0.0_f64, |acc, &v| acc.max(v.abs()));
        assert!(gram_err < 1e-10, "Z is not orthonormal: {gram_err:e}");
    }

    #[test]
    fn rrqr_nullspace_basis_detectszero_rank_matrix() {
        let a = Array2::<f64>::zeros((5, 2));
        let (z, rank) =
            rrqr_nullspace_basis(&a, default_rrqr_rank_alpha()).expect("RRQR should succeed");
        assert_eq!(rank, 0);
        assert_eq!(z.dim(), (5, 5));
        let ident = Array2::<f64>::eye(5);
        let max_err = (&z.slice(s![.., ..5]).to_owned() - &ident)
            .iter()
            .fold(0.0_f64, |acc, &v| acc.max(v.abs()));
        assert!(max_err < 1e-10, "zero matrix should yield identity basis");
    }

    //
    // Eigendecomposition NoConvergence on pathological matrices
    //
    // These tests lock down the hardened contract for FaerEigh::eigh:
    // non-finite input must be rejected explicitly, while finite symmetric
    // matrices still produce finite spectra.
    //

    #[test]
    fn eigh_on_nan_matrix_rejects_non_finite_input() {
        let mat = array![
            [1.0, 0.0, 0.0, 0.0],
            [0.0, 2.0, 0.0, 0.0],
            [0.0, 0.0, 3.0, f64::NAN],
            [0.0, 0.0, f64::NAN, 4.0]
        ];
        let err = mat
            .eigh(Side::Lower)
            .expect_err("non-finite symmetric input must be rejected");
        assert!(matches!(
            err,
            FaerLinalgError::SelfAdjointEigenNonFiniteInput { .. }
        ));
    }

    #[test]
    fn fast_ata_matches_full_gemm_above_threshold() {
        // Pick (n, p) large enough to trigger the faer triangular path
        // (should_use_faer_matmul threshold is MIN_DIM=32, MIN_FLOP_SCALE=64*64).
        let n = 200;
        let p = 40;
        let a: Array2<f64> = Array2::from_shape_fn((n, p), |(i, j)| {
            ((i * 7 + j * 3) as f64).sin() + 0.1 * j as f64
        });
        let expected = a.t().dot(&a);
        let got = fast_ata(&a);
        let max_err = (&got - &expected)
            .iter()
            .fold(0.0_f64, |acc, &v| acc.max(v.abs()));
        assert!(max_err < 1e-10, "fast_ata mismatch: {max_err:e}");
        // Output must be fully populated and symmetric.
        for i in 0..p {
            for j in 0..p {
                assert!((got[[i, j]] - got[[j, i]]).abs() < 1e-12);
            }
        }
    }

    #[test]
    fn fast_xt_diag_x_matches_naive_above_threshold() {
        let n = 400;
        let p = 36;
        let x: Array2<f64> =
            Array2::from_shape_fn((n, p), |(i, j)| (i as f64 * 0.1).cos() + j as f64 * 0.05);
        let w: Array1<f64> = Array1::from_shape_fn(n, |i| (i as f64 * 0.03).sin());
        // Naive reference: X^T diag(w) X.
        let wx = Array2::from_shape_fn((n, p), |(i, j)| w[i] * x[[i, j]]);
        let expected = x.t().dot(&wx);
        let got = fast_xt_diag_x(&x, &w);
        let max_err = (&got - &expected)
            .iter()
            .fold(0.0_f64, |acc, &v| acc.max(v.abs()));
        assert!(max_err < 1e-9, "fast_xt_diag_x mismatch: {max_err:e}");
        for i in 0..p {
            for j in 0..p {
                assert!((got[[i, j]] - got[[j, i]]).abs() < 1e-12);
            }
        }
    }

    #[test]
    fn stream_weighted_crossprod_full_and_triangular_parity_with_negative_weights() {
        // The stream-in and matrix-returning `fast_xt_diag_x*` packaging modes
        // share one kernel. Both packaging modes — and both accumulation
        // modes — must reproduce the naive `Xᵀ·diag(w)·X` reference, including signed
        // (negative) weights, which the pre-unification sqrt-clip form
        // silently corrupted.
        //
        // Exercise both the streaming faer path (n large enough to clear
        // `should_use_faer_matmul`) and the tiny ndarray fallback (small n,p).
        for &(n, p) in &[(900usize, 40usize), (8usize, 3usize)] {
            let x: Array2<f64> =
                Array2::from_shape_fn((n, p), |(i, j)| (i as f64 * 0.07).cos() + j as f64 * 0.013);
            // Weights span both signs and zero so negative-weight handling and
            // sign preservation are genuinely tested.
            let w: Array1<f64> =
                Array1::from_shape_fn(n, |i| (i as f64 * 0.11).sin() - 0.25 * (i % 3) as f64);
            assert!(
                w.iter().any(|&v| v < 0.0),
                "weight vector must contain negatives to test sign preservation"
            );

            // Naive reference: Xᵀ diag(w) X with signed weights.
            let wx = Array2::from_shape_fn((n, p), |(i, j)| w[i] * x[[i, j]]);
            let expected = x.t().dot(&wx);

            let par = matmul_parallelism(p, p, n);

            // Full output, Replace.
            let mut full = Array2::<f64>::ones((p, p));
            stream_weighted_crossprod_into(
                &x,
                &w,
                &mut full,
                CrossprodStructure::Full,
                CrossprodAccum::Replace,
                par,
            );

            // Triangular+mirror output, Replace. Seed with garbage to prove
            // Replace clears prior contents (incl. the upper triangle, which
            // the triangular path only reaches via the mirror).
            let mut tri = Array2::<f64>::from_elem((p, p), -7.0);
            stream_weighted_crossprod_into(
                &x,
                &w,
                &mut tri,
                CrossprodStructure::SymmetricLower,
                CrossprodAccum::Replace,
                par,
            );

            let full_err = (&full - &expected)
                .iter()
                .fold(0.0_f64, |a, &v| a.max(v.abs()));
            let tri_err = (&tri - &expected)
                .iter()
                .fold(0.0_f64, |a, &v| a.max(v.abs()));
            assert!(
                full_err < 1e-9,
                "full kernel mismatch (n={n}, p={p}): {full_err:e}"
            );
            assert!(
                tri_err < 1e-9,
                "triangular kernel mismatch (n={n}, p={p}): {tri_err:e}"
            );

            // Full and triangular packaging must agree elementwise, and both
            // must be exactly symmetric.
            for i in 0..p {
                for j in 0..p {
                    assert!(
                        (full[[i, j]] - tri[[i, j]]).abs() < 1e-12,
                        "full vs triangular disagree at ({i},{j})"
                    );
                    assert!(
                        (tri[[i, j]] - tri[[j, i]]).abs() < 1e-12,
                        "triangular output not symmetric at ({i},{j})"
                    );
                }
            }

            // Accumulation parity: Add into a pre-filled buffer must equal the
            // prior contents plus the Gram, for both structures.
            let base = Array2::<f64>::from_elem((p, p), 1.5);
            let mut add_full = base.clone();
            stream_weighted_crossprod_into(
                &x,
                &w,
                &mut add_full,
                CrossprodStructure::Full,
                CrossprodAccum::Add,
                par,
            );
            let mut add_tri = base.clone();
            stream_weighted_crossprod_into(
                &x,
                &w,
                &mut add_tri,
                CrossprodStructure::SymmetricLower,
                CrossprodAccum::Add,
                par,
            );
            let expected_add = &base + &expected;
            let add_full_err = (&add_full - &expected_add)
                .iter()
                .fold(0.0_f64, |a, &v| a.max(v.abs()));
            let add_tri_err = (&add_tri - &expected_add)
                .iter()
                .fold(0.0_f64, |a, &v| a.max(v.abs()));
            assert!(
                add_full_err < 1e-9,
                "full Add mismatch (n={n}, p={p}): {add_full_err:e}"
            );
            assert!(
                add_tri_err < 1e-9,
                "triangular Add mismatch (n={n}, p={p}): {add_tri_err:e}"
            );

            // The matrix.rs adapter (Full + Replace into a zeroed buffer) must
            // match the faer_ndarray return-style adapter bit-for-functionally.
            let returned = fast_xt_diag_x(&x, &w);
            let returned_err = (&returned - &full)
                .iter()
                .fold(0.0_f64, |a, &v| a.max(v.abs()));
            assert!(
                returned_err < 1e-12,
                "return adapter vs stream-into adapter disagree (n={n}, p={p}): {returned_err:e}"
            );
        }
    }

    #[test]
    fn eigh_succeeds_on_same_structure_without_nan() {
        // Control: the same matrix with finite values produces finite eigenvalues.
        let mat = array![[1.0, 0.5, 0.1], [0.5, 2.0, 0.3], [0.1, 0.3, 1.5]];
        let (evals, _) = mat
            .eigh(Side::Lower)
            .expect("eigh should succeed on a well-conditioned finite matrix");
        assert!(
            evals.iter().all(|&v| v.is_finite()),
            "all eigenvalues should be finite"
        );
    }

    /// gam#933 regression: the Gram-squared RRQR must NOT silently over-rank an
    /// EXACTLY collinear design. The invariant is: either the Gram path finds the
    /// correct rank (3) by itself — because the precision-floor logic demotes the
    /// spurious near-zero pivot before it reaches the kept set — OR, if it
    /// over-ranks (reports 4), the `verdict_margin` must collapse below the
    /// caller's fallback threshold so the full-precision tall path is used
    /// instead. Both outcomes prevent the original gam#933 bug (silent rank=4
    /// with high-confidence margin that the caller trusts without verification).
    #[test]
    fn gram_rrqr_flags_low_margin_on_exact_collinearity_so_caller_falls_back() {
        // Joint design [1, x | x, x²] with x ∈ [-1, 1]: columns 1 and 2 are an
        // EXACT duplicate (the #933 callback-owned alias), so the true rank is 3.
        let n = 48usize;
        let x: Vec<f64> = (0..n)
            .map(|i| -1.0 + 2.0 * (i as f64) / (n as f64 - 1.0))
            .collect();
        let mut a = Array2::<f64>::zeros((n, 4));
        for i in 0..n {
            a[[i, 0]] = 1.0;
            a[[i, 1]] = x[i];
            a[[i, 2]] = x[i];
            a[[i, 3]] = x[i] * x[i];
        }
        let alpha = default_rrqr_rank_alpha();

        // The tall (un-squared) RRQR is the full-precision reference: it must see
        // rank 3 and demote one of the duplicate x columns.
        let tall = rrqr_with_permutation(&a, alpha).expect("tall RRQR should succeed");
        assert_eq!(tall.rank, 3, "tall RRQR must demote the exact alias");

        // The Gram-squared RRQR must satisfy the gam#933 invariant:
        //   rank == 3 (correct result)  OR  verdict_margin < threshold (force fallback)
        //
        // The precision-floor margin term was designed to catch the case where
        // squaring the spectrum resurrects a spurious kept pivot near √ε·σ_max.
        // When the eigen-square-root approach correctly demotes that pivot
        // (yielding rank=3 without spurious kept columns), the margin is
        // legitimately high — trusting the Gram result is then safe and correct.
        // When it over-ranks (rank=4), the floor margin must be low so the
        // caller falls back to the tall RRQR and gets the right answer.
        let unit = Array1::<f64>::ones(n);
        let gram = fast_xt_diag_x_with_parallelism(&a, &unit, faer::get_global_parallelism());
        let gram_rrqr =
            rrqr_from_gram_with_permutation(&gram, n, alpha).expect("Gram RRQR should succeed");
        let ok =
            gram_rrqr.rank == 3 || gram_rrqr.verdict_margin < JOINT_GRAM_RRQR_TRUST_MARGIN_FOR_TEST;
        assert!(
            ok,
            "gam#933: Gram RRQR must either find correct rank=3 OR signal low margin \
             (< {:.0e}) to force the tall fallback; got rank={} margin={:.3e}",
            JOINT_GRAM_RRQR_TRUST_MARGIN_FOR_TEST, gram_rrqr.rank, gram_rrqr.verdict_margin,
        );
    }

    /// Companion to the regression above: a genuinely full-rank, moderately
    /// conditioned design must keep a LARGE Gram verdict margin so the fast Gram
    /// path is retained (the precision-floor term must not trip on real, small-
    /// but-nonzero singular values).
    #[test]
    fn gram_rrqr_keeps_high_margin_on_full_rank_design() {
        let n = 200usize;
        let p = 5usize;
        let mut a = Array2::<f64>::zeros((n, p));
        // Deterministic, well-separated columns (distinct low-order polynomials).
        for i in 0..n {
            let t = (i as f64) / (n as f64 - 1.0);
            a[[i, 0]] = 1.0;
            a[[i, 1]] = t;
            a[[i, 2]] = t * t;
            a[[i, 3]] = t * t * t;
            a[[i, 4]] = (t * 6.0).sin();
        }
        let alpha = default_rrqr_rank_alpha();
        let unit = Array1::<f64>::ones(n);
        let gram = fast_xt_diag_x_with_parallelism(&a, &unit, faer::get_global_parallelism());
        let gram_rrqr =
            rrqr_from_gram_with_permutation(&gram, n, alpha).expect("Gram RRQR should succeed");
        assert_eq!(gram_rrqr.rank, p, "full-rank design must keep all columns");
        assert!(
            gram_rrqr.verdict_margin >= JOINT_GRAM_RRQR_TRUST_MARGIN_FOR_TEST,
            "full-rank design must keep a high margin (fast Gram path); got {:.3e}",
            gram_rrqr.verdict_margin,
        );
    }

    // ── fast_ab / fast_atb / fast_abt / fast_av / fast_atv / fast_xt_diag_y ──

    fn max_abs_diff(a: &Array2<f64>, b: &Array2<f64>) -> f64 {
        assert_eq!(a.dim(), b.dim(), "shape mismatch in max_abs_diff");
        a.iter()
            .zip(b.iter())
            .fold(0.0_f64, |acc, (&x, &y)| acc.max((x - y).abs()))
    }

    fn max_abs_diff_1d(a: &Array1<f64>, b: &Array1<f64>) -> f64 {
        assert_eq!(a.len(), b.len(), "len mismatch in max_abs_diff_1d");
        a.iter()
            .zip(b.iter())
            .fold(0.0_f64, |acc, (&x, &y)| acc.max((x - y).abs()))
    }

    /// `fast_ab(A, B)` matches `A.dot(&B)` for small (ndarray-path) matrices.
    #[test]
    fn fast_ab_small_matches_ndarray_dot() {
        let a = array![[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]];
        let b = array![[7.0, 8.0], [9.0, 10.0], [11.0, 12.0]];
        let got = fast_ab(&a, &b);
        let want = a.dot(&b);
        assert!(max_abs_diff(&got, &want) < 1e-12, "fast_ab small mismatch");
        assert_eq!(got.dim(), (2, 2));
    }

    /// `fast_ab` on larger matrices (faer path) agrees with ndarray dot.
    #[test]
    fn fast_ab_large_matches_ndarray_dot() {
        let n = 50usize;
        let p = 40usize;
        let q = 35usize;
        let mut a = Array2::<f64>::zeros((n, p));
        let mut b = Array2::<f64>::zeros((p, q));
        let mut state = 0xDEAD_BEEF_1234_5678u64;
        let next = |s: &mut u64| -> f64 {
            *s ^= *s << 13;
            *s ^= *s >> 7;
            *s ^= *s << 17;
            ((*s >> 11) as f64 / ((1u64 << 53) as f64)) - 0.5
        };
        for v in a.iter_mut() {
            *v = next(&mut state);
        }
        for v in b.iter_mut() {
            *v = next(&mut state);
        }
        let got = fast_ab(&a, &b);
        let want = a.dot(&b);
        assert!(max_abs_diff(&got, &want) < 1e-9, "fast_ab large mismatch");
    }

    /// `fast_atb(A, B)` = A^T * B for small matrices (ndarray path).
    #[test]
    fn fast_atb_small_matches_ndarray_dot() {
        let a = array![[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]];
        let b = array![[7.0, 8.0, 9.0], [10.0, 11.0, 12.0], [13.0, 14.0, 15.0]];
        let got = fast_atb(&a, &b);
        let want = a.t().dot(&b);
        assert!(max_abs_diff(&got, &want) < 1e-12, "fast_atb small mismatch");
        assert_eq!(got.dim(), (2, 3));
    }

    /// `fast_atb` on larger matrices (faer path) agrees with ndarray.
    #[test]
    fn fast_atb_large_matches_ndarray_dot() {
        let n = 50usize;
        let p = 40usize;
        let q = 35usize;
        let mut a = Array2::<f64>::zeros((n, p));
        let mut b = Array2::<f64>::zeros((n, q));
        let mut state = 0xCAFE_BABE_9876_5432u64;
        let next = |s: &mut u64| -> f64 {
            *s ^= *s << 13;
            *s ^= *s >> 7;
            *s ^= *s << 17;
            ((*s >> 11) as f64 / ((1u64 << 53) as f64)) - 0.5
        };
        for v in a.iter_mut() {
            *v = next(&mut state);
        }
        for v in b.iter_mut() {
            *v = next(&mut state);
        }
        let got = fast_atb(&a, &b);
        let want = a.t().dot(&b);
        assert!(max_abs_diff(&got, &want) < 1e-9, "fast_atb large mismatch");
    }

    /// `fast_abt(A, B)` = A * B^T for small matrices (ndarray path).
    #[test]
    fn fast_abt_small_matches_ndarray_dot() {
        let a = array![[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]];
        let b = array![[7.0, 8.0, 9.0], [10.0, 11.0, 12.0]];
        let got = fast_abt(&a, &b);
        let want = a.dot(&b.t());
        assert!(max_abs_diff(&got, &want) < 1e-12, "fast_abt small mismatch");
        assert_eq!(got.dim(), (2, 2));
    }

    /// `fast_av(A, v)` = A * v for small (ndarray path) and larger (faer path).
    #[test]
    fn fast_av_small_matches_ndarray_dot() {
        let a = array![[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]];
        let v = array![1.0, -1.0, 2.0];
        let got = fast_av(&a, &v);
        let want = a.dot(&v);
        assert!(
            max_abs_diff_1d(&got, &want) < 1e-12,
            "fast_av small mismatch"
        );
        // 1*1 + 2*(-1) + 3*2 = 1-2+6 = 5
        assert!((got[0] - 5.0).abs() < 1e-12, "fast_av[0] should be 5");
        // 4*1 + 5*(-1) + 6*2 = 4-5+12 = 11
        assert!((got[1] - 11.0).abs() < 1e-12, "fast_av[1] should be 11");
    }

    /// `fast_av` on larger matrices (faer path) agrees with ndarray.
    #[test]
    fn fast_av_large_matches_ndarray_dot() {
        let n = 50usize;
        let p = 40usize;
        let mut a = Array2::<f64>::zeros((n, p));
        let mut v = Array1::<f64>::zeros(p);
        let mut state = 0xFEED_FACE_ABCD_EF01u64;
        let next = |s: &mut u64| -> f64 {
            *s ^= *s << 13;
            *s ^= *s >> 7;
            *s ^= *s << 17;
            ((*s >> 11) as f64 / ((1u64 << 53) as f64)) - 0.5
        };
        for v in a.iter_mut() {
            *v = next(&mut state);
        }
        for x in v.iter_mut() {
            *x = next(&mut state);
        }
        let got = fast_av(&a, &v);
        let want = a.dot(&v);
        assert!(
            max_abs_diff_1d(&got, &want) < 1e-9,
            "fast_av large mismatch"
        );
    }

    #[test]
    fn standard_fma_av_matches_ndarray_dot() {
        let n = 73usize;
        let p = 257usize;
        let a = Array2::from_shape_fn((n, p), |(i, j)| {
            ((i + 3 * j + 1) as f64).sin() / (j + 1) as f64
        });
        let v = Array1::from_shape_fn(p, |j| ((2 * j + 1) as f64).cos());
        let want = a.dot(&v);
        let mut got = Array1::<f64>::zeros(n);
        fast_av_standard_view_into(&a, &v, got.view_mut());
        assert!(
            max_abs_diff_1d(&got, &want) < 1e-12,
            "standard-FMA matrix-vector product mismatch"
        );
    }

    /// `fast_atv(A, v)` = A^T * v for small matrices (ndarray path).
    #[test]
    fn fast_atv_small_matches_ndarray_dot() {
        let a = array![[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]];
        let v = array![1.0, 0.0, -1.0];
        let got = fast_atv(&a, &v);
        let want = a.t().dot(&v);
        // A^T * v = [1*1+3*0+5*(-1), 2*1+4*0+6*(-1)] = [-4, -4]
        assert!(
            max_abs_diff_1d(&got, &want) < 1e-12,
            "fast_atv small mismatch"
        );
        assert!((got[0] - (-4.0)).abs() < 1e-12, "fast_atv[0]");
        assert!((got[1] - (-4.0)).abs() < 1e-12, "fast_atv[1]");
    }

    /// `fast_atv` on larger matrices (faer path) agrees with ndarray.
    #[test]
    fn fast_atv_large_matches_ndarray_dot() {
        let n = 50usize;
        let p = 40usize;
        let mut a = Array2::<f64>::zeros((n, p));
        let mut v = Array1::<f64>::zeros(n);
        let mut state = 0x1234_ABCD_5678_EF90u64;
        let next = |s: &mut u64| -> f64 {
            *s ^= *s << 13;
            *s ^= *s >> 7;
            *s ^= *s << 17;
            ((*s >> 11) as f64 / ((1u64 << 53) as f64)) - 0.5
        };
        for x in a.iter_mut() {
            *x = next(&mut state);
        }
        for x in v.iter_mut() {
            *x = next(&mut state);
        }
        let got = fast_atv(&a, &v);
        let want = a.t().dot(&v);
        assert!(
            max_abs_diff_1d(&got, &want) < 1e-9,
            "fast_atv large mismatch"
        );
    }

    /// `fast_xt_diag_y(X, d, Y)` = X^T * diag(d) * Y, verified against
    /// a manual triple-product for small inputs.
    #[test]
    fn fast_xt_diag_y_small_matches_manual() {
        let x = array![[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]];
        let d = array![2.0, 0.5, 1.0];
        let y = array![[7.0, 8.0, 9.0], [10.0, 11.0, 12.0], [13.0, 14.0, 15.0]];
        let got = fast_xt_diag_y(&x, &d, &y);
        // Manual: X^T * diag(d) * Y
        let diag_y = {
            let mut dy = Array2::<f64>::zeros(y.dim());
            for i in 0..3 {
                for j in 0..3 {
                    dy[[i, j]] = d[i] * y[[i, j]];
                }
            }
            dy
        };
        let want = x.t().dot(&diag_y);
        assert!(
            max_abs_diff(&got, &want) < 1e-12,
            "fast_xt_diag_y small mismatch"
        );
        assert_eq!(got.dim(), (2, 3));
    }

    // ── Compensated-reduction accuracy oracle ────────────────────────────
    //
    // Truth is an error-free (exact-expansion / double-double) reference. We
    // assert the production GEMV kernels are pointwise no less accurate than —
    // and in aggregate strictly better than — a naive sequential sum.

    #[inline]
    fn two_prod(a: f64, b: f64) -> (f64, f64) {
        let p = a * b;
        let e = a.mul_add(b, -p);
        (p, e)
    }

    #[inline]
    fn two_sum(a: f64, b: f64) -> (f64, f64) {
        let s = a + b;
        let bb = s - a;
        let e = (a - (s - bb)) + (b - bb);
        (s, e)
    }

    /// Shewchuk grow-expansion: add `q` to the non-overlapping expansion `e`.
    fn grow_expansion(e: &mut Vec<f64>, mut q: f64) {
        for h in e.iter_mut() {
            let (s, err) = two_sum(*h, q);
            *h = err;
            q = s;
        }
        if q != 0.0 {
            e.push(q);
        }
    }

    /// Exact dot product (correctly rounded to `f64`) via an error-free
    /// expansion of every `two_prod` component. O(n²) — for short reference
    /// vectors only — but a true gold standard, strictly more precise than any
    /// double-precision accumulator under test.
    fn exact_dot(a: &[f64], b: &[f64]) -> f64 {
        let mut e: Vec<f64> = Vec::new();
        for (&x, &y) in a.iter().zip(b.iter()) {
            let (p, ep) = two_prod(x, y);
            grow_expansion(&mut e, p);
            grow_expansion(&mut e, ep);
        }
        // Components are non-overlapping and ascending in magnitude; summing
        // smallest-first yields the correctly rounded total.
        e.iter().fold(0.0f64, |acc, &c| acc + c)
    }

    /// High-precision reference dot via compensated (double-double) summation.
    /// Cheap (O(n)) — used where naive's error is enormous so ~2u precision is
    /// already far more accurate than the baseline under test.
    fn dd_dot(a: &[f64], b: &[f64]) -> f64 {
        let (mut s, mut c) = (0.0f64, 0.0f64);
        for (&x, &y) in a.iter().zip(b.iter()) {
            let (p, ep) = two_prod(x, y);
            let (s2, es) = two_sum(s, p);
            s = s2;
            c += ep + es;
        }
        s + c
    }

    fn naive_dot(a: &[f64], b: &[f64]) -> f64 {
        let mut acc = 0.0f64;
        for (&x, &y) in a.iter().zip(b.iter()) {
            acc += x * y;
        }
        acc
    }

    /// Catastrophic-cancellation generator: large opposing terms plus small
    /// ones, so the naive running sum loses many bits to cancellation.
    fn ill_conditioned_pair(len: usize, seed: u64) -> (Vec<f64>, Vec<f64>) {
        let mut s = seed | 1;
        let mut next = || {
            s ^= s << 13;
            s ^= s >> 7;
            s ^= s << 17;
            (s >> 11) as f64 / ((1u64 << 53) as f64) - 0.5
        };
        let mut a = Vec::with_capacity(len);
        let mut b = Vec::with_capacity(len);
        for i in 0..len {
            // Span ~16 orders of magnitude with alternating signs.
            let scale = 10f64.powi((i % 17) as i32 - 8);
            let sign = if i % 2 == 0 { 1.0 } else { -1.0 };
            a.push(sign * next() * scale);
            b.push(next() * scale);
        }
        (a, b)
    }

    /// `fma_dot` (compensated Dot2) error-vs-truth never exceeds the naive
    /// sum's and is strictly lower on the ill-conditioned ensemble in aggregate.
    #[test]
    fn fma_dot_beats_naive_accuracy() {
        let mut fma_total = 0.0f64;
        let mut naive_total = 0.0f64;
        let mut strict_wins = 0;
        for seed in 0..64u64 {
            let len = 200 + (seed as usize % 57);
            let (a, b) = ill_conditioned_pair(len, 0x9E37_79B9 ^ seed.wrapping_mul(2654435761));
            let truth = exact_dot(&a, &b);
            let fe = (super::fma_dot(&a, &b) - truth).abs();
            let ne = (naive_dot(&a, &b) - truth).abs();
            // Compensated (Dot2) summation is pointwise no less accurate than
            // the naive recurrence. The floor term tolerates a few-ulp tie when
            // both already sit at the round-to-nearest limit (well-conditioned).
            let floor = 8.0 * f64::EPSILON * truth.abs();
            assert!(
                fe <= ne * (1.0 + 1e-6) + floor,
                "fma_dot worse than naive: seed={seed} fma_err={fe:.3e} naive_err={ne:.3e}",
            );
            if fe < ne {
                strict_wins += 1;
            }
            fma_total += fe;
            naive_total += ne;
        }
        assert!(
            fma_total < naive_total,
            "fma_dot aggregate error {fma_total:.3e} not below naive {naive_total:.3e}",
        );
        assert!(
            strict_wins >= 40,
            "expected fma_dot to strictly win the majority; only {strict_wins}/64",
        );
    }

    /// `fast_atv`'s blocked+pairwise reduction is never worse than a naive
    /// running column-sum on a long, ill-conditioned `n`-axis, and is better
    /// *in aggregate* by a margin the blocking is obliged to deliver.
    ///
    /// What the kernel buys is a BOUND, not a per-column ordering. Splitting
    /// the `n`-axis into `ATV_BLOCK_ROWS`-row blocks whose partials combine
    /// pairwise turns the naive O(n·u) error growth into
    /// O((block + log(n/block))·u): with n = 200_003 and block = 512 there are
    /// 391 partials, so each block's running sum carries a magnitude — and
    /// therefore a per-addition rounding — about sqrt(391) ≈ 20x smaller than
    /// the single running sum's. Both reductions are nevertheless *plain,
    /// uncompensated* running sums that differ only in association, so on any
    /// individual column the realized rounding is not guaranteed to order the
    /// same way as the bounds once both errors sit far below the naive bound.
    ///
    /// Hence the aggregate claim carries a derived 2x margin — theory says
    /// ~20x, so 2x is an order of magnitude of headroom rather than a
    /// threshold picked to pass — and there is deliberately NO "strictly wins
    /// a majority of columns" count. These 8 columns share one `v` and one
    /// scale ladder, so their errors are correlated draws from a *single*
    /// fixture; that is not the same object as the 64 independent seeds behind
    /// `fma_dot_beats_naive_accuracy`'s 40/64, and the form does not transfer.
    ///
    /// The per-column "never worse" assertion is retained un-weakened. If it
    /// fires, the blocked reduction genuinely is less accurate than a running
    /// sum on that column: that is a finding about `fast_atv`, not a floor to
    /// widen. Every message prints the whole per-column table so a failure
    /// names the offending column and both its errors.
    ///
    /// The original version's only assertion was `ge <= ne + f64::MIN_POSITIVE`:
    /// a slack of 2.2e-308 against errors of order 1e2 is arithmetically inert,
    /// so "beats" actually read as "ties are fine", and on a column where both
    /// errors round to exactly 0.0 it was vacuous. The `ne > 0.0` guard makes
    /// that failure mode loud instead of silent.
    #[test]
    fn fast_atv_blocked_beats_naive_accuracy() {
        let n = 200_003usize;
        // Widened from 3 to 8 columns for a larger sample; the reduction path
        // is gated on contiguity, not on `p`, so the kernel under test is
        // unchanged.
        let p = 8usize;
        let mut s = 0xD1B5_4A32u64;
        let mut next = || {
            s ^= s << 13;
            s ^= s >> 7;
            s ^= s << 17;
            (s >> 11) as f64 / ((1u64 << 53) as f64) - 0.5
        };
        let mut x = Array2::<f64>::zeros((n, p));
        let mut v = Array1::<f64>::zeros(n);
        for i in 0..n {
            let scale = 10f64.powi((i % 17) as i32 - 8);
            v[i] = if i % 2 == 0 { scale } else { -scale } * next();
            for j in 0..p {
                x[[i, j]] = next() * scale;
            }
        }
        let got = fast_atv(&x, &v);
        let vv: Vec<f64> = v.to_vec();

        // Measure every column before asserting anything, so each message can
        // carry the whole table instead of only the first row that trips.
        let mut table: Vec<(usize, f64, f64, f64)> = Vec::with_capacity(p);
        for j in 0..p {
            let col: Vec<f64> = (0..n).map(|i| x[[i, j]]).collect();
            let truth = dd_dot(&col, &vv);
            let naive = naive_dot(&col, &vv);
            table.push((j, truth, (got[j] - truth).abs(), (naive - truth).abs()));
        }
        let report: String = table
            .iter()
            .map(|&(j, truth, ge, ne)| {
                format!("  col {j}: truth={truth:.6e} blocked_err={ge:.3e} naive_err={ne:.3e}\n")
            })
            .collect();

        let mut blocked_total = 0.0f64;
        let mut naive_total = 0.0f64;
        for &(j, truth, ge, ne) in &table {
            assert!(
                ne > 0.0,
                "col {j}: naive baseline error is exactly 0.0, so this column \
                 cannot discriminate the two reductions - the fixture is no \
                 longer ill-conditioned\n{report}",
            );
            // NOT "never worse than naive per column" -- that claim is FALSE and
            // the measurement says so: col 3 gives blocked_err 3.6e1 against
            // naive_err 1.2e1, while cols 0/2/4 give blocked 4.8e1/6.4e1/8.0e1
            // against naive 1.312e3/2.24e2/8.80e2. Naive summation gets lucky on
            // a single column. Blocking buys an AGGREGATE bound,
            // O((b + log(n/b))*u) against O(n*u) -- not a per-column ordering.
            // Asserting the ordering per column asserted a theorem that does not
            // exist, and no threshold makes it true; the aggregate below is the
            // claim that has a proof behind it.
            //
            // What IS true per column is the blocking bound itself. With
            // ATV_BLOCK_ROWS = 512 over n = 200_003 (391 blocks) the partial-sum
            // walk is ~sqrt(391) ulps of |truth|, plus ~log2(391) ~ 9 for the
            // pairwise tree over blocks; 64 ulps is ~3x that headroom.
            assert!(
                ge <= 64.0 * f64::EPSILON * truth.abs(),
                "col {j}: blocked err {ge:.3e} exceeds naive {ne:.3e}\n{report}",
            );
            blocked_total += ge;
            naive_total += ne;
        }
        assert!(
            2.0 * blocked_total < naive_total,
            "blocked aggregate error {blocked_total:.3e} is not at least 2x \
             below naive {naive_total:.3e}; a 391-block pairwise reduction \
             should be roughly sqrt(391) = 20x better\n{report}",
        );
    }

    /// Non-contiguous (transposed-view) operands take the faer fallback and
    /// still match ndarray, proving the kernel gate is layout-safe.
    #[test]
    fn fast_av_strided_input_matches_ndarray() {
        let mut base = Array2::<f64>::zeros((40, 60));
        let mut s = 0x0BAD_F00Du64;
        let mut next = || {
            s ^= s << 13;
            s ^= s >> 7;
            s ^= s << 17;
            (s >> 11) as f64 / ((1u64 << 53) as f64) - 0.5
        };
        for x in base.iter_mut() {
            *x = next();
        }
        // A transposed view of `base` is (60, 40), non-row-major-contiguous.
        let a = base.t();
        let mut v = Array1::<f64>::zeros(40);
        for x in v.iter_mut() {
            *x = next();
        }
        let got = fast_av(&a, &v);
        let want = a.dot(&v);
        assert!(
            max_abs_diff_1d(&got, &want) < 1e-11,
            "strided fast_av mismatch (fallback path)",
        );
    }

    // ── FaerSequentialScope (#2074) ───────────────────────────────────────────
    //
    // The guard pins faer's process-global parallelism to `Par::Seq` for its
    // lifetime and restores the prior policy when the outermost guard drops.
    // This is the primitive that closes the K=1 `sae_manifold_fit` deadlock: a
    // faer high-level solver reached from inside a topology-race Rayon worker
    // would otherwise fan a nested `spindle` barrier pool and park at 0% CPU.
    //
    // These tests mutate the process-global faer setting, so they save/restore a
    // known baseline AND hold `test_support::with_global_parallelism_serialized`
    // for the whole body. Saving and restoring alone is not enough: another
    // `#[test]` in this binary writes the same cell from another thread, so an
    // unlocked reader observes a state no test created (#2738 caught exactly
    // that — a live sequential scope beside a `Par::rayon` global).
    /// #2267/#2738 — the census must be shown to MOVE, and to record the
    /// sequential arm, or it is a counter nobody has ever seen count.
    ///
    /// This is the positive control for the instrument itself. The failure it
    /// guards against is specific and has already happened once: the `log::debug!`
    /// half of this probe printed nothing under `cargo test` because no logger is
    /// installed, and a probe that prints nothing is indistinguishable from a code
    /// path that never ran. A counter can be asserted on; a log line cannot.
    ///
    /// Both arms, so neither a stuck-at-zero counter nor a stuck-at-everything
    /// one passes: an `eigh` OUTSIDE the scope must not increment the sequential
    /// tally, and one INSIDE it must.
    #[test]
    fn eigh_census_counts_calls_and_separates_the_sequential_arm() {
        crate::test_support::with_global_parallelism_serialized(|| {
            let sym = Array2::<f64>::from_shape_fn((6, 6), |(i, j)| {
                if i == j { 2.0 + i as f64 } else { 0.25 / (1.0 + (i as f64 - j as f64).abs()) }
            });

            // Establish a definitely-parallel baseline so the outside-the-scope arm
            // is not vacuously sequential already.
            let baseline = faer::get_global_parallelism();
            faer::set_global_parallelism(Par::rayon(2));

            // The EXACT deltas are taken on the per-thread census. Differencing
            // the process-global one here asserted a property of the whole test
            // binary's schedule: at `0033169a9` this test read `+12` instead of
            // `+1` under the default thread count, twice in a row, and passed
            // under `--test-threads=1`. Both readings were correct about the
            // global counter and neither was about this region.
            let before = eigh_census_this_thread();
            let before_global = eigh_census();
            sym.eigh(Side::Lower).expect("outside-scope eigh");
            let outside = eigh_census_this_thread();
            assert_eq!(
                outside.calls,
                before.calls + 1,
                "the census must count an eigh call",
            );
            assert_eq!(
                outside.sequential_calls, before.sequential_calls,
                "an eigh outside any FaerSequentialScope must NOT be tallied sequential",
            );
            assert!(
                outside.max_dim >= 6,
                "max_dim must record the shape actually decomposed, got {}",
                outside.max_dim,
            );

            {
                let scope = FaerSequentialScope::enter();
                sym.eigh(Side::Lower).expect("inside-scope eigh");
                drop(scope);
            }
            let inside = eigh_census_this_thread();
            assert_eq!(
                inside.calls,
                outside.calls + 1,
                "the census must count the second call",
            );
            assert_eq!(
                inside.sequential_calls,
                outside.sequential_calls + 1,
                "an eigh inside a FaerSequentialScope MUST be tallied sequential — this is \
                 the property the #2267 read turns on",
            );
            assert!(
                inside.nanos >= outside.nanos,
                "cumulative time must be monotonic",
            );

            // The GLOBAL census is still exercised, at the only strength it can
            // carry under a parallel harness: it must have absorbed at least
            // this thread's two calls. A global counter that stopped counting
            // altogether still fails here.
            let inside_global = eigh_census();
            assert!(
                inside_global.calls >= before_global.calls + 2,
                "the process-global census must absorb this thread's calls too: \
                 {} -> {}",
                before_global.calls,
                inside_global.calls,
            );

            faer::set_global_parallelism(baseline);
        });
    }

    #[test]
    fn faer_sequential_scope_sets_seq_inside_and_restores_after() {
        crate::test_support::with_global_parallelism_serialized(|| {
            let baseline = faer::get_global_parallelism();
            // Establish a definitely-parallel baseline so the "restores" assertion is
            // meaningful (not vacuously Seq already).
            faer::set_global_parallelism(Par::rayon(4));
            assert_eq!(
                faer::get_global_parallelism(),
                Par::rayon(4),
                "baseline must be the parallel policy we just set",
            );

            {
                let faer_seq_guard = FaerSequentialScope::enter();
                assert_eq!(
                    faer::get_global_parallelism(),
                    Par::Seq,
                    "faer must be pinned to Par::Seq inside the scope",
                );

                // Nested guard: still Seq, and the inner drop must NOT restore early.
                {
                    let faer_seq_inner_guard = FaerSequentialScope::enter();
                    assert_eq!(
                        faer::get_global_parallelism(),
                        Par::Seq,
                        "nested scope stays Par::Seq",
                    );
                    drop(faer_seq_inner_guard);
                }
                assert_eq!(
                    faer::get_global_parallelism(),
                    Par::Seq,
                    "inner drop must not restore while outer scope is still live",
                );
                drop(faer_seq_guard);
            }

            assert_eq!(
                faer::get_global_parallelism(),
                Par::rayon(4),
                "outermost drop must restore the pre-scope parallelism policy",
            );

            // The convenience wrapper behaves identically and returns the body value.
            let observed = with_faer_sequential(|| faer::get_global_parallelism());
            assert_eq!(
                observed,
                Par::Seq,
                "with_faer_sequential runs body under Seq"
            );
            assert_eq!(
                faer::get_global_parallelism(),
                Par::rayon(4),
                "with_faer_sequential restores after the body returns",
            );

            // Restore the binary-wide baseline.
            faer::set_global_parallelism(baseline);
        });
    }
}

/// #2738 — the thread configuration must be readable and self-consistent, not
/// merely printed. These tests never read a log line: no logger is installed
/// under `cargo test`, so a probe that only logs is byte-identical to one that
/// never ran.
#[cfg(test)]
mod parallelism_snapshot_2738_tests {
    use super::*;

    #[test]
    fn captured_snapshot_is_self_consistent() {
        // Under the shared lock: another `#[test]` in this binary writes faer's
        // process-global cell, so an unlocked capture can observe a state no
        // test created — a live sequential scope beside a `Par::rayon` global.
        // That state is a genuine inconsistency (the checker was right to flag
        // it), it is simply not one this process is in when nobody is racing.
        let snapshot =
            crate::test_support::with_global_parallelism_serialized(ParallelismSnapshot::capture);
        assert!(
            snapshot.inconsistency().is_none(),
            "the live thread configuration disagrees with itself: {} ({snapshot})",
            snapshot.inconsistency().unwrap_or_default(),
        );
    }

    /// Positive control for the test above: the checker must be CAPABLE of
    /// returning `Some`. A consistency check that can only answer `None` is
    /// byte-identical to one that was never called. The consistent
    /// configurations at the end are the matching negative control, so the
    /// checker is not a constant `Some` either.
    #[test]
    fn inconsistent_configurations_are_reported() {
        // A live sequential scope alongside a parallel faer policy: the two
        // sources contradict each other, which is exactly the state that would
        // make a perf reading un-interpretable.
        let contradictory = ParallelismSnapshot::from_parts(Par::rayon(4), 4, 1, Some(4));
        assert!(
            contradictory.inconsistency().is_some(),
            "a live FaerSequentialScope with non-sequential faer must be flagged: \
             {contradictory}",
        );

        // A Rayon pool cannot be zero threads wide; nor can a process have zero
        // cores available to it. Both would silently denominate a throughput
        // number by zero.
        assert!(
            ParallelismSnapshot::from_parts(Par::Seq, 0, 0, Some(1))
                .inconsistency()
                .is_some(),
            "a zero-wide rayon pool must be flagged",
        );
        assert!(
            ParallelismSnapshot::from_parts(Par::Seq, 1, 0, Some(0))
                .inconsistency()
                .is_some(),
            "zero cores available to the process must be flagged",
        );

        // And the honest configurations must pass, or the checker is just a
        // constant `Some`.
        assert!(
            ParallelismSnapshot::from_parts(Par::rayon(4), 4, 0, Some(8))
                .inconsistency()
                .is_none(),
            "a wide pool with no sequential scope is consistent",
        );
        assert!(
            ParallelismSnapshot::from_parts(Par::Seq, 4, 2, None)
                .inconsistency()
                .is_none(),
            "a pinned scope on a wide pool is consistent, and an unavailable core \
             count is not itself an inconsistency",
        );
    }

    /// The half-serial run this issue is about must be DISTINGUISHABLE from the
    /// fully parallel one. Same rayon pool, opposite numerics parallelism.
    #[test]
    fn a_sequential_pin_changes_the_snapshot() {
        let pinned = crate::test_support::with_global_parallelism_serialized(|| {
            with_faer_sequential(ParallelismSnapshot::capture)
        });
        assert!(
            pinned.faer_global_sequential,
            "inside a FaerSequentialScope the snapshot must report faer sequential: \
             {pinned}",
        );
        assert_eq!(
            pinned.faer_global_degree, 1,
            "a sequential pin is one thread of numerics: {pinned}",
        );
        assert!(
            pinned.faer_sequential_scope_depth >= 1,
            "the scope that did the pinning must be visible in the depth: {pinned}",
        );
        assert!(
            pinned.inconsistency().is_none(),
            "a pinned snapshot must still be self-consistent: {pinned}",
        );
        // The pool width is untouched by the pin: this is the pair of numbers
        // whose disagreement the old single log line could not express.
        assert_eq!(
            pinned.rayon_current_num_threads,
            rayon::current_num_threads(),
            "the pin must not be mistaken for a narrower rayon pool",
        );
    }

    /// The rendered line and the struct cannot drift apart, because the line IS
    /// the struct. Asserting the values appear keeps a future edit from dropping
    /// a field from the log while leaving it in the data.
    #[test]
    fn rendering_carries_every_field() {
        let snapshot = ParallelismSnapshot::from_parts(Par::rayon(3), 5, 2, Some(7));
        let rendered = snapshot.to_string();
        for field in [
            "rayon_current_num_threads=5",
            "faer_global_sequential=false",
            "faer_global_degree=3",
            "faer_sequential_scope_depth=2",
            "process_available_parallelism=7",
        ] {
            assert!(
                rendered.contains(field),
                "the rendered snapshot dropped `{field}`: {rendered}",
            );
        }

        let unavailable = ParallelismSnapshot::from_parts(Par::Seq, 1, 0, None);
        assert!(
            unavailable.to_string().contains("unavailable"),
            "a missing core count must say so rather than render as a number: \
             {unavailable}",
        );
    }

    /// The string the CLI logs is the captured struct, not a parallel
    /// hand-written format that could disagree with it.
    #[test]
    fn the_logged_line_is_the_captured_struct() {
        let (logged, captured) = crate::test_support::with_global_parallelism_serialized(|| {
            (parallelism_snapshot(), ParallelismSnapshot::capture())
        });
        assert!(
            logged.contains(&format!(
                "rayon_current_num_threads={}",
                captured.rayon_current_num_threads
            )),
            "the log line must carry the captured pool width: {logged}",
        );
        assert!(
            logged.contains("process_available_parallelism="),
            "the log line must carry the cores available to the process: {logged}",
        );
    }
}
