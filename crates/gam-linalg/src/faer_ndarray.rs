use dyn_stack::{MemBuffer, MemStack};
use faer::diag::{Diag, DiagRef};
use faer::linalg::solvers;
use faer::linalg::svd::{self, ComputeSvdVectors};
use faer::perm::{Perm, PermRef};
use faer::prelude::ReborrowMut;
use faer::{Conj, Mat, MatMut, MatRef, Par, Side, get_global_parallelism};
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
/// `Par::Seq` via `pool_parallelism` instead of re-fanning the
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
/// Deliberately NOT reported: `pool_parallelism`. That is
/// per call and only governs the codebase's own products and LLTs; it cannot
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

/// #2627 — the degree a matrix product, a Gram, or an LLT factorization runs at,
/// passed to faer per call: the width of the pool the call runs in, or `Par::Seq`
/// once a `NestedParallelGuard` or a [`FaerSequentialScope`] is live, so a product
/// issued from inside a parallel fan-out does not multiply the live thread count
/// against the outer pool or wait on a `spindle` barrier the outer pool holds.
///
/// These kernels give identical bits at every degree and every pool width (GEMM
/// 2000, LLT 2000 and Gram 200k×100 at degrees Seq/4/8/16/24/48 on pools of
/// 1/4/24 threads; pool job 1120486), so the degree moves only the wall time.
#[inline]
pub fn pool_parallelism() -> Par {
    if in_nested_parallel_region() || faer_sequential_scope_depth() > 0 {
        Par::Seq
    } else {
        Par::rayon(0)
    }
}

/// #2627 — the degree every decomposition and triangular solve runs at, except
/// the self-adjoint EVD ([`evd_parallelism`]): sequential, passed to faer per
/// call.
///
/// A decomposition's arithmetic partition follows its degree — thin QR and SVD
/// split at `min(degree, K, 4)`, the self-adjoint EVD at every degree ≥ 2 (pool
/// jobs 1101442 and 1120486) — so a degree read from the pool made a fit a
/// function of `RAYON_NUM_THREADS`. On one core sequential is also the fastest
/// degree; at production sizes it costs 1.2× on QR (50k×200) and nothing on
/// the SVD.
#[inline]
pub fn decomposition_parallelism() -> Par {
    Par::Seq
}

/// #2627/#2267 — the degree every self-adjoint eigendecomposition is
/// partitioned at: the named deployment cap [`EVD_DEPLOYMENT_DEGREE`], passed
/// to faer per call.
///
/// **Invariance.** faer's EVD words follow the degree its arithmetic is
/// partitioned at, not the number of threads that execute it, so the degree has
/// to be a constant of the call and never a runtime quantity such as the pool
/// width (`Par::rayon(0)`). At n = 7692 degree 24 gave identical words on pools
/// of width 24, 8 and 1 on EPYC 9534 (job 1161259), EPYC 7702 (job 1159471) and
/// EPYC 7763 (job 1180582), which also shows identical words across pools at
/// n = 1500 and 3000 for every fixed degree it measured.
#[inline]
pub fn evd_parallelism() -> Par {
    Par::rayon(EVD_DEPLOYMENT_DEGREE)
}

/// #2627/#2267 — the degree the self-adjoint EVD is partitioned at: a named
/// deployment choice, not a tolerance.
///
/// **Why it is a constant.** Eigen words must be identical across pool widths,
/// so the degree cannot depend on the pool. faer 0.24's tridiagonal reduction
/// (`linalg::evd::tridiag::tridiag_in_place`) splits the trailing symmetric
/// matvec into exactly `degree` column chunks of equal triangular area. It
/// falls back to sequential only once `rem²/2 < par_threshold = 192·256`. The
/// split count is the degree at every dimension, so no size saturates it. The
/// one pool-independent degree faer's structure offers is the one whose chunks
/// each carry faer's grain at the first reduction step,
/// `max(1, ⌊n²/(2·192·256)⌋)`. It loses to task-spawn overhead at production
/// size (job 1180582, EPYC 7763, faer 0.24.4, median wall time):
///
/// | n    | pool | degree 24 | grain degree        |
/// |------|------|-----------|---------------------|
/// | 3000 | 24   | 3.55 s    | 4.76 s (degree 91)  |
/// | 7692 | 8    | 42.7 s    | 76.1 s (degree 601) |
/// | 7692 | 24   | 27.8 s    | 79.6 s (degree 601) |
///
/// **Why 24.** Measured on EPYC 7763 with faer 0.24.4 (job 1180582), at n = 7692,
/// the dense exact-A dimension of the manifold-SAE criterion: 4.67× faster than
/// sequential on a pool of 24 threads, 3.03× on 8, and +0.3% on one thread
/// (129.9 s against 129.5 s). Earlier cells: +3.5% at n = 1024 and +2.6% at
/// n = 1500 on one thread (job 1161804), and 8.5× and 5.7× on 24 and 8 threads on
/// EPYC 9534 (job 1161259). A pool wider than 24 runs the EVD at 24.
///
/// **A deployment choice.** Nothing numerical depends on the value; the words
/// pin and `evd_deployment_degree_reaches_faer_as_its_chunk_count_2627` gate it.
/// Re-measure it whenever faer is bumped.
const EVD_DEPLOYMENT_DEGREE: usize = 24;

/// Process-global depth counter + saved parallelism for [`FaerSequentialScope`].
///
/// gam-linalg's own factorizations and products take their degree per call
/// ([`decomposition_parallelism`], [`pool_parallelism`]) and never read faer's
/// global policy. What this scope still reaches is faer's high-level
/// factorization/solve entry points called directly outside gam-linalg (`thin_svd`,
/// `self_adjoint_eigen`, `col_piv_qr().solve_lstsq`, the sparse LLT), which read
/// `faer::get_global_parallelism()` internally and have no per-call parallelism
/// argument, and [`pool_parallelism`], which it collapses to `Par::Seq`. When such a solver runs from inside a
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
/// reaches stays single-threaded and never spawns a nested barrier pool. The swap
/// is bit-for-bit neutral only for degree-invariant kernels (products, Grams,
/// LLT). A decomposition still called through faer's high-level solvers gives
/// different bits under it (#2627), which is why gam-linalg's decompositions run
/// at [`decomposition_parallelism`] per call instead.
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
pub(crate) fn faer_sequential_scope_depth() -> usize {
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
    #[error("General eigendecomposition failed: {0:?}")]
    GeneralEigen(solvers::EvdError),
    #[error(
        "General eigendecomposition certificate refused: {arm} at slot {slot} measured {measured:.3e} against band {band:.3e}"
    )]
    GeneralEigenCertificateRefused { arm: &'static str, slot: usize, measured: f64, band: f64 },
    #[error("Cholesky factorization failed: {0:?}")]
    Cholesky(solvers::LltError),
    #[error("LDLT factorization failed: {0:?}")]
    Ldlt(solvers::LdltError),
}

/// The lower triangle of the symmetric matrix `a` whose stored triangle `side`
/// names.
fn lower_triangle(a: MatRef<'_, f64>, side: Side) -> Mat<f64> {
    let n = a.nrows();
    let mut lower = Mat::<f64>::zeros(n, n);
    match side {
        Side::Lower => lower.copy_from_triangular_lower(a),
        Side::Upper => lower.copy_from_triangular_lower(a.transpose()),
    }
    lower
}

/// Zero everything above the diagonal of `a`.
fn zero_strict_upper(mut a: MatMut<'_, f64>) {
    for j in 0..a.ncols() {
        for i in 0..j.min(a.nrows()) {
            a[(i, j)] = 0.0;
        }
    }
}

/// #2627 — `A = L Lᵀ` of a symmetric positive-definite matrix, owned by gam
/// rather than borrowed from `faer::linalg::solvers`. faer's high-level solvers
/// read faer's PROCESS-GLOBAL parallelism, whose default under the `rayon` feature
/// is `Par::rayon(0)`: the width of whatever pool the call runs in. This type calls
/// faer's low-level entry points with the degree passed per call —
/// [`pool_parallelism`] for the factorization, whose bits are degree-invariant, and
/// [`decomposition_parallelism`] for the triangular solves — so gam never reads or
/// sets faer's global policy, and never leaks one into another faer user.
#[derive(Clone)]
pub struct FaerLlt<T> {
    l: Mat<T>,
}

impl FaerLlt<f64> {
    /// Factor the symmetric matrix `a` from the triangle `side` names.
    pub fn new(a: MatRef<'_, f64>, side: Side) -> Result<Self, solvers::LltError> {
        let mut l = lower_triangle(a, side);
        let par = pool_parallelism();
        let n = l.nrows();
        let mut mem = MemBuffer::new(
            faer::linalg::cholesky::llt::factor::cholesky_in_place_scratch::<f64>(
                n,
                par,
                Default::default(),
            ),
        );
        faer::linalg::cholesky::llt::factor::cholesky_in_place(
            l.as_mut(),
            Default::default(),
            par,
            MemStack::new(&mut mem),
            Default::default(),
        )?;
        zero_strict_upper(l.as_mut());
        Ok(Self { l })
    }

    /// The lower-triangular factor `L`.
    pub fn lower(&self) -> MatRef<'_, f64> {
        self.l.as_ref()
    }

    /// The dimension of the factored matrix.
    pub fn nrows(&self) -> usize {
        self.l.nrows()
    }

    /// Overwrite `rhs` with `A⁻¹ rhs`.
    pub fn solve_in_place(&self, rhs: MatMut<'_, f64>) {
        let par = decomposition_parallelism();
        let mut mem = MemBuffer::new(
            faer::linalg::cholesky::llt::solve::solve_in_place_scratch::<f64>(
                self.l.nrows(),
                rhs.ncols(),
                par,
            ),
        );
        faer::linalg::cholesky::llt::solve::solve_in_place_with_conj(
            self.l.as_ref(),
            Conj::No,
            rhs,
            par,
            MemStack::new(&mut mem),
        );
    }

    /// `A⁻¹ rhs`.
    pub fn solve(&self, rhs: MatRef<'_, f64>) -> Mat<f64> {
        let mut out = rhs.to_owned();
        self.solve_in_place(out.as_mut());
        out
    }
}

/// `A = L D Lᵀ` of a symmetric matrix at [`decomposition_parallelism`].
#[derive(Clone)]
pub struct FaerLdlt {
    l: Mat<f64>,
    d: Diag<f64>,
}

impl FaerLdlt {
    /// Factor the symmetric matrix `a` from the triangle `side` names.
    pub fn new(a: MatRef<'_, f64>, side: Side) -> Result<Self, solvers::LdltError> {
        let mut l = lower_triangle(a, side);
        let par = decomposition_parallelism();
        let n = l.nrows();
        let mut mem = MemBuffer::new(
            faer::linalg::cholesky::ldlt::factor::cholesky_in_place_scratch::<f64>(
                n,
                par,
                Default::default(),
            ),
        );
        faer::linalg::cholesky::ldlt::factor::cholesky_in_place(
            l.as_mut(),
            Default::default(),
            par,
            MemStack::new(&mut mem),
            Default::default(),
        )?;
        let mut d = Diag::<f64>::zeros(n);
        d.copy_from(l.diagonal());
        l.diagonal_mut().fill(1.0);
        zero_strict_upper(l.as_mut());
        Ok(Self { l, d })
    }

    /// The diagonal factor `D`.
    pub fn diagonal(&self) -> DiagRef<'_, f64> {
        self.d.as_ref()
    }

    /// The dimension of the factored matrix.
    pub fn nrows(&self) -> usize {
        self.l.nrows()
    }

    /// Overwrite `rhs` with `A⁻¹ rhs`.
    pub fn solve_in_place(&self, rhs: MatMut<'_, f64>) {
        let par = decomposition_parallelism();
        let mut mem = MemBuffer::new(
            faer::linalg::cholesky::ldlt::solve::solve_in_place_scratch::<f64>(
                self.l.nrows(),
                rhs.ncols(),
                par,
            ),
        );
        faer::linalg::cholesky::ldlt::solve::solve_in_place_with_conj(
            self.l.as_ref(),
            self.d.as_ref(),
            Conj::No,
            rhs,
            par,
            MemStack::new(&mut mem),
        );
    }

    /// `A⁻¹ rhs`.
    pub fn solve(&self, rhs: MatRef<'_, f64>) -> Mat<f64> {
        let mut out = rhs.to_owned();
        self.solve_in_place(out.as_mut());
        out
    }
}

/// The inertia of a symmetric matrix read off its Bunch–Kaufman factor (#2901).
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct SymmetricInertia {
    pub negative: usize,
    pub zero: usize,
    pub positive: usize,
    /// The smallest eigenvalue of the factor's 1×1 and 2×2 pivot blocks: NaN when any
    /// pivot is not finite, `+∞` for an empty matrix.
    pub smallest_pivot: f64,
    /// `‖L̂‖_F²·‖B̂‖_∞`, a bound on `‖ |L̂||B̂||L̂ᵀ| ‖₂`, the factors' term in the
    /// factorization's backward error.
    pub factor_magnitude: f64,
}

/// `P A Pᵀ = L B Lᵀ` (Bunch-Kaufman) of a symmetric matrix at
/// [`decomposition_parallelism`].
#[derive(Clone)]
pub struct FaerLblt {
    l: Mat<f64>,
    b_diag: Diag<f64>,
    b_subdiag: Diag<f64>,
    perm: Perm<usize>,
}

impl FaerLblt {
    /// Factor the symmetric matrix `a` from the triangle `side` names.
    pub fn new(a: MatRef<'_, f64>, side: Side) -> Self {
        let mut l = lower_triangle(a, side);
        let par = decomposition_parallelism();
        let n = l.nrows();
        let mut b_subdiag = Diag::<f64>::zeros(n);
        let mut forward = vec![0usize; n];
        let mut inverse = vec![0usize; n];
        let mut mem = MemBuffer::new(
            faer::linalg::cholesky::lblt::factor::cholesky_in_place_scratch::<usize, f64>(
                n,
                par,
                Default::default(),
            ),
        );
        faer::linalg::cholesky::lblt::factor::cholesky_in_place(
            l.as_mut(),
            b_subdiag.as_mut(),
            &mut forward,
            &mut inverse,
            par,
            MemStack::new(&mut mem),
            Default::default(),
        );
        let mut b_diag = Diag::<f64>::zeros(n);
        b_diag.copy_from(l.diagonal());
        l.diagonal_mut().fill(1.0);
        zero_strict_upper(l.as_mut());
        let perm = Perm::new_checked(forward.into_boxed_slice(), inverse.into_boxed_slice(), n);
        Self {
            l,
            b_diag,
            b_subdiag,
            perm,
        }
    }

    /// The inertia of the factored matrix, read off `B` (#2901).
    ///
    /// `P A Pᵀ = L B Lᵀ` with `L` unit lower triangular is a congruence, so by
    /// Sylvester's law of inertia `A` has as many negative, zero and positive
    /// eigenvalues as `B`. `B` is block diagonal: a nonzero subdiagonal entry at `k`
    /// opens a 2×2 block on `k, k + 1`, and every other index is a 1×1 block. The
    /// eigenvalues of `B` are those of its blocks.
    pub fn inertia(&self) -> SymmetricInertia {
        let n = self.l.nrows();
        let b_diag = self.b_diag.as_ref();
        let b_subdiag = self.b_subdiag.as_ref();
        let mut inertia = SymmetricInertia {
            negative: 0,
            zero: 0,
            positive: 0,
            smallest_pivot: f64::INFINITY,
            factor_magnitude: 0.0,
        };
        let mut finite = true;
        let mut block_norm = 0.0_f64;
        let mut count = |value: f64, inertia: &mut SymmetricInertia| {
            if !value.is_finite() {
                finite = false;
            } else if value < 0.0 {
                inertia.negative += 1;
            } else if value > 0.0 {
                inertia.positive += 1;
            } else {
                inertia.zero += 1;
            }
            inertia.smallest_pivot = inertia.smallest_pivot.min(value);
        };
        let mut k = 0;
        while k < n {
            let a = b_diag[k];
            if k + 1 < n && b_subdiag[k] != 0.0 {
                let c = b_subdiag[k];
                let d = b_diag[k + 1];
                let mean = 0.5 * (a + d);
                let radius = (0.5 * (a - d)).hypot(c);
                count(mean - radius, &mut inertia);
                count(mean + radius, &mut inertia);
                block_norm = block_norm.max((a.abs() + c.abs()).max(c.abs() + d.abs()));
                k += 2;
            } else {
                count(a, &mut inertia);
                block_norm = block_norm.max(a.abs());
                k += 1;
            }
        }
        if !finite {
            inertia.smallest_pivot = f64::NAN;
        }
        let mut lower_frobenius_sq = 0.0_f64;
        for j in 0..n {
            for i in j..n {
                lower_frobenius_sq += self.l[(i, j)] * self.l[(i, j)];
            }
        }
        inertia.factor_magnitude = lower_frobenius_sq * block_norm;
        inertia
    }

    /// The dimension of the factored matrix.
    pub fn nrows(&self) -> usize {
        self.l.nrows()
    }

    /// Overwrite `rhs` with `A⁻¹ rhs`.
    pub fn solve_in_place(&self, rhs: MatMut<'_, f64>) {
        let par = decomposition_parallelism();
        let mut mem = MemBuffer::new(
            faer::linalg::cholesky::lblt::solve::solve_in_place_scratch::<usize, f64>(
                self.l.nrows(),
                rhs.ncols(),
                par,
            ),
        );
        faer::linalg::cholesky::lblt::solve::solve_in_place_with_conj(
            self.l.as_ref(),
            self.b_diag.as_ref(),
            self.b_subdiag.as_ref(),
            Conj::No,
            self.perm.as_ref(),
            rhs,
            par,
            MemStack::new(&mut mem),
        );
    }

    /// `A⁻¹ rhs`.
    pub fn solve(&self, rhs: MatRef<'_, f64>) -> Mat<f64> {
        let mut out = rhs.to_owned();
        self.solve_in_place(out.as_mut());
        out
    }
}

/// Householder QR `A = Q R` at [`decomposition_parallelism`]: the unit lower
/// trapezoidal reflector basis (`m × min(m, n)`), its block coefficients, and the
/// upper trapezoidal `R` (`min(m, n) × n`).
fn householder_qr(a: MatRef<'_, f64>) -> (Mat<f64>, Mat<f64>, Mat<f64>) {
    let (m, n) = a.shape();
    let size = m.min(n);
    let par = decomposition_parallelism();
    let mut qr = a.to_owned();
    let block_size = faer::linalg::qr::no_pivoting::factor::recommended_block_size::<f64>(m, n);
    let mut coeff = Mat::<f64>::zeros(block_size, size);
    let mut mem = MemBuffer::new(faer::linalg::qr::no_pivoting::factor::qr_in_place_scratch::<f64>(
        m,
        n,
        block_size,
        par,
        Default::default(),
    ));
    faer::linalg::qr::no_pivoting::factor::qr_in_place(
        qr.as_mut(),
        coeff.as_mut(),
        par,
        MemStack::new(&mut mem),
        Default::default(),
    );
    let (basis, r) = split_householder(qr.as_ref());
    (basis, coeff, r)
}

/// Split a factored `QR` buffer into the unit lower trapezoidal reflector basis
/// and the upper trapezoidal `R`.
fn split_householder(qr: MatRef<'_, f64>) -> (Mat<f64>, Mat<f64>) {
    let (m, n) = qr.shape();
    let size = m.min(n);
    let mut r = Mat::<f64>::zeros(size, n);
    for j in 0..n {
        for i in 0..size.min(j + 1) {
            r[(i, j)] = qr[(i, j)];
        }
    }
    let mut basis = Mat::<f64>::zeros(m, size);
    for j in 0..size {
        basis[(j, j)] = 1.0;
        for i in j + 1..m {
            basis[(i, j)] = qr[(i, j)];
        }
    }
    (basis, r)
}

/// Apply the reflector sequence `Q` of a Householder factorization to `target`
/// from the left at [`decomposition_parallelism`].
fn apply_householder_on_the_left(basis: MatRef<'_, f64>, coeff: MatRef<'_, f64>, target: MatMut<'_, f64>) {
    let columns = target.ncols();
    faer::linalg::householder::apply_block_householder_sequence_on_the_left_in_place_with_conj(
        basis,
        coeff,
        Conj::No,
        target,
        decomposition_parallelism(),
        MemStack::new(&mut MemBuffer::new(
            faer::linalg::householder::apply_block_householder_sequence_on_the_left_in_place_scratch::<f64>(
                basis.nrows(),
                coeff.nrows(),
                columns,
            ),
        )),
    );
}

/// Householder QR `A = Q R` at [`decomposition_parallelism`], keeping the
/// reflectors so `Qᵀ` applies to a right-hand side without `Q` being formed.
pub struct HouseholderQr {
    basis: Mat<f64>,
    coeff: Mat<f64>,
    r: Mat<f64>,
}

impl HouseholderQr {
    pub fn new(a: MatRef<'_, f64>) -> Self {
        let (basis, coeff, r) = householder_qr(a);
        Self { basis, coeff, r }
    }

    /// The upper trapezoidal `R`, `min(m, n) × n`.
    pub fn r(&self) -> MatRef<'_, f64> {
        self.r.as_ref()
    }

    /// `target ← Qᵀ target`, at [`decomposition_parallelism`].
    pub fn apply_transpose_on_the_left(&self, target: MatMut<'_, f64>) {
        let columns = target.ncols();
        faer::linalg::householder::apply_block_householder_sequence_transpose_on_the_left_in_place_with_conj(
            self.basis.as_ref(),
            self.coeff.as_ref(),
            Conj::No,
            target,
            decomposition_parallelism(),
            MemStack::new(&mut MemBuffer::new(
                faer::linalg::householder::apply_block_householder_sequence_transpose_on_the_left_in_place_scratch::<f64>(
                    self.basis.nrows(),
                    self.coeff.nrows(),
                    columns,
                ),
            )),
        );
    }
}

/// Column-pivoted Householder QR `A P = Q R` at [`decomposition_parallelism`].
struct ColumnPivotedQr {
    basis: Mat<f64>,
    coeff: Mat<f64>,
    r: Mat<f64>,
    forward: Vec<usize>,
    inverse: Vec<usize>,
}

impl ColumnPivotedQr {
    fn new(a: MatRef<'_, f64>) -> Self {
        let (m, n) = a.shape();
        let size = m.min(n);
        let par = decomposition_parallelism();
        let mut qr = a.to_owned();
        let block_size = faer::linalg::qr::no_pivoting::factor::recommended_block_size::<f64>(m, n);
        let mut coeff = Mat::<f64>::zeros(block_size, size);
        let mut forward = vec![0usize; n];
        let mut inverse = vec![0usize; n];
        let mut mem = MemBuffer::new(
            faer::linalg::qr::col_pivoting::factor::qr_in_place_scratch::<usize, f64>(
                m,
                n,
                block_size,
                par,
                Default::default(),
            ),
        );
        faer::linalg::qr::col_pivoting::factor::qr_in_place(
            qr.as_mut(),
            coeff.as_mut(),
            &mut forward,
            &mut inverse,
            par,
            MemStack::new(&mut mem),
            Default::default(),
        );
        let (basis, r) = split_householder(qr.as_ref());
        Self {
            basis,
            coeff,
            r,
            forward,
            inverse,
        }
    }

    /// The upper trapezoidal `R`, `min(m, n) × n`.
    fn thin_r(&self) -> MatRef<'_, f64> {
        self.r.as_ref()
    }
}

/// #2627 — the least-squares solution `X = argmin ‖A X − B‖` of a tall `A`
/// (`nrows ≥ ncols`) by column-pivoted Householder QR at
/// [`decomposition_parallelism`], replacing `col_piv_qr().solve_lstsq`, which
/// reads faer's process-global parallelism.
pub fn col_piv_qr_solve_lstsq(a: MatRef<'_, f64>, rhs: MatRef<'_, f64>) -> Mat<f64> {
    let (m, n) = a.shape();
    let qr = ColumnPivotedQr::new(a);
    let par = decomposition_parallelism();
    let columns = rhs.ncols();
    let mut out = rhs.to_owned();
    faer::linalg::qr::col_pivoting::solve::solve_lstsq_in_place_with_conj(
        qr.basis.as_ref(),
        qr.coeff.as_ref(),
        qr.r.as_ref(),
        PermRef::new_checked(&qr.forward, &qr.inverse, n),
        Conj::No,
        out.as_mut(),
        par,
        MemStack::new(&mut MemBuffer::new(
            faer::linalg::qr::col_pivoting::solve::solve_lstsq_in_place_scratch::<usize, f64>(
                m,
                n,
                qr.coeff.nrows(),
                columns,
                par,
            ),
        )),
    );
    out.truncate(n, columns);
    out
}

pub enum FaerSymmetricFactor {
    Llt(FaerLlt<f64>),
    Ldlt(FaerLdlt),
    Lblt(FaerLblt),
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
            FaerSymmetricFactor::Llt(f) => cholesky_factor_logdet(f.lower()),
            FaerSymmetricFactor::Ldlt(f) => diagonal_log_sum(f.diagonal()),
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
        // `pool_parallelism` collapses to `Par::Seq` when this GEMM is reached
        // from inside a `NestedParallelGuard` row region, preventing the
        // Rayon-pool × faer-pool multiplicative oversubscription.
        pool_parallelism()
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
//     (`f64::mul_add` is only an instruction when the code is COMPILED with
//     the `fma` target feature; the portable x86_64 baseline this workspace
//     ships has none, so a plain build lowered every `mul_add` here to a
//     call into the runtime `fma` dispatcher — measured on a gaussian
//     n=50,000 / p=93 fit: zero `vfmadd` in the row-major matvec closure and
//     28% of the fit's cycles inside `fma`/`fma_with_fma`. Each kernel below
//     is therefore compiled twice, once for the baseline and once with
//     `fma,avx2` enabled, and its entry point picks the second whenever the
//     running CPU reports both features — see `fma_avx2_available`.)
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
fn fma_dot_body(a: &[f64], b: &[f64]) -> f64 {
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

/// Whether the running x86_64 CPU executes the `fma,avx2` kernel variants.
///
/// `is_x86_feature_detected!` caches its probe in a process-wide static, so
/// this is a load and a bit test per call — invisible next to any kernel whose
/// row is at least [`FMA_LANES`] long.
#[cfg(target_arch = "x86_64")]
#[inline]
fn fma_avx2_available() -> bool {
    std::arch::is_x86_feature_detected!("fma") && std::arch::is_x86_feature_detected!("avx2")
}

/// [`fma_dot_body`] compiled with the `fma,avx2` target features, so its
/// `mul_add`s are `vfmadd` instructions over packed lanes instead of calls.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "fma,avx2")]
fn fma_dot_fma_avx2(a: &[f64], b: &[f64]) -> f64 {
    fma_dot_body(a, b)
}

/// Compensated dot product: [`fma_dot_body`] on the CPU-feature variant the
/// running machine supports. Bit-identical across variants — an FMA is an FMA
/// whether it is an instruction or the runtime library's implementation of
/// one, and the eight lanes keep their per-lane order under vectorization.
#[inline]
fn fma_dot(a: &[f64], b: &[f64]) -> f64 {
    #[cfg(target_arch = "x86_64")]
    if fma_avx2_available() {
        // SAFETY: `fma_avx2_available` is the cached CPU probe for exactly the
        // `fma` and `avx2` features this variant enables.
        return unsafe { fma_dot_fma_avx2(a, b) };
    }
    fma_dot_body(a, b)
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
fn standard_fma_dot_body(a: &[f64], b: &[f64]) -> f64 {
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

/// [`standard_fma_dot_body`] compiled with the `fma,avx2` target features.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "fma,avx2")]
fn standard_fma_dot_fma_avx2(a: &[f64], b: &[f64]) -> f64 {
    standard_fma_dot_body(a, b)
}

/// Ordinary FMA dot product on the CPU-feature variant the running machine
/// supports (bit-identical across variants, as for [`fma_dot`]).
#[inline]
fn standard_fma_dot(a: &[f64], b: &[f64]) -> f64 {
    #[cfg(target_arch = "x86_64")]
    if fma_avx2_available() {
        // SAFETY: `fma_avx2_available` is the cached CPU probe for exactly the
        // `fma` and `avx2` features this variant enables.
        return unsafe { standard_fma_dot_fma_avx2(a, b) };
    }
    standard_fma_dot_body(a, b)
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
        atv_block_accumulate(&x_all[start * p..end * p], &v[start..end], &mut acc);
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

/// `acc[j] += Σ_i rows[i,j]·v[i]` over a row-major block `rows` of `v.len()`
/// rows and `acc.len()` columns: the private partial of one
/// [`fast_atv_rowmajor_into`] reduction block, one FMA per entry.
#[inline(always)]
fn atv_block_accumulate_body(rows: &[f64], v: &[f64], acc: &mut [f64]) {
    let p = acc.len();
    assert_eq!(rows.len(), v.len() * p, "atv_block_accumulate: block length");
    for (&vi, row) in v.iter().zip(rows.chunks_exact(p)) {
        for (a, &xij) in acc.iter_mut().zip(row.iter()) {
            *a = xij.mul_add(vi, *a);
        }
    }
}

/// [`atv_block_accumulate_body`] compiled with the `fma,avx2` target features.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "fma,avx2")]
fn atv_block_accumulate_fma_avx2(rows: &[f64], v: &[f64], acc: &mut [f64]) {
    atv_block_accumulate_body(rows, v, acc)
}

/// Block partial of `Xᵀv` on the CPU-feature variant the running machine
/// supports (bit-identical across variants, as for [`fma_dot`]).
#[inline]
fn atv_block_accumulate(rows: &[f64], v: &[f64], acc: &mut [f64]) {
    #[cfg(target_arch = "x86_64")]
    if fma_avx2_available() {
        // SAFETY: `fma_avx2_available` is the cached CPU probe for exactly the
        // `fma` and `avx2` features this variant enables.
        return unsafe { atv_block_accumulate_fma_avx2(rows, v, acc) };
    }
    atv_block_accumulate_body(rows, v, acc)
}

/// `y[i] = alpha·x[i] + y[i]` with one FMA per entry, on the CPU-feature
/// variant the running machine supports (bit-identical across variants, as
/// for the compensated dot above). The
/// Lanczos reorthogonalization's projection update is this kernel over the
/// full basis at every step, so it pays the same per-`mul_add` call price as
/// the matvecs without it.
pub(crate) fn fma_axpy_into(alpha: f64, x: &[f64], y: &mut [f64]) {
    #[cfg(target_arch = "x86_64")]
    if fma_avx2_available() {
        // SAFETY: `fma_avx2_available` is the cached CPU probe for exactly the
        // `fma` and `avx2` features this variant enables.
        return unsafe { fma_axpy_into_fma_avx2(alpha, x, y) };
    }
    fma_axpy_into_body(alpha, x, y)
}

#[inline(always)]
fn fma_axpy_into_body(alpha: f64, x: &[f64], y: &mut [f64]) {
    assert_eq!(x.len(), y.len(), "fma_axpy_into: operand length mismatch");
    for (yi, &xi) in y.iter_mut().zip(x.iter()) {
        *yi = alpha.mul_add(xi, *yi);
    }
}

/// [`fma_axpy_into_body`] compiled with the `fma,avx2` target features.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "fma,avx2")]
fn fma_axpy_into_fma_avx2(alpha: f64, x: &[f64], y: &mut [f64]) {
    fma_axpy_into_body(alpha, x, y)
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
pub(crate) fn fast_av_standard_view_into<S1: Data<Elem = f64>, S2: Data<Elem = f64>>(
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
pub(crate) fn fast_atv_into<S1: Data<Elem = f64>, S2: Data<Elem = f64>>(
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
    // A product the matmul dispatcher keeps sequential stays sequential whatever
    // policy the caller passes. faer's parallel split of a small GEMM changes its
    // summation order with the pool width, so a small fit's bits would follow
    // `RAYON_NUM_THREADS` (#2627 thread_count_reproducibility).
    let par = if matmul_parallelism(p, p, n) == Par::Seq {
        Par::Seq
    } else {
        par
    };

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

pub(crate) struct FaerColView<'a> {
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
            let par = decomposition_parallelism();
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

        let par = decomposition_parallelism();
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

/// Self-adjoint eigendecomposition `A = U diag(S) Uᵀ` of the triangle `side`
/// names, at [`evd_parallelism`].
pub fn self_adjoint_evd(a: MatRef<'_, f64>, side: Side) -> Result<(Diag<f64>, Mat<f64>), solvers::EvdError> {
    let n = a.nrows();
    let par = evd_parallelism();
    let lower = match side {
        Side::Lower => a,
        Side::Upper => a.transpose(),
    };
    let mut s = Diag::<f64>::zeros(n);
    let mut u = Mat::<f64>::zeros(n, n);
    let mut mem = MemBuffer::new(faer::linalg::evd::self_adjoint_evd_scratch::<f64>(
        n,
        faer::linalg::evd::ComputeEigenvectors::Yes,
        par,
        Default::default(),
    ));
    faer::linalg::evd::self_adjoint_evd(
        lower,
        s.as_mut(),
        Some(u.as_mut()),
        par,
        MemStack::new(&mut mem),
        Default::default(),
    )?;
    Ok((s, u))
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
    let (s, u) = catch_unwind(AssertUnwindSafe(|| self_adjoint_evd(view.as_ref(), side)))
        .map_err(|_| FaerLinalgError::FactorizationFailed {
            context: "strict self-adjoint eigendecomposition panic boundary",
        })?
        .map_err(FaerLinalgError::SelfAdjointEigen)?;
    let values = diag_to_array(s.as_ref());
    let vectors = mat_to_array(u.as_ref());
    if values.iter().any(|value| !value.is_finite())
        || vectors.iter().any(|value| !value.is_finite())
    {
        return Err(FaerLinalgError::SelfAdjointEigenNonFiniteInput {
            context: "strict self-adjoint eigendecomposition output validation",
        });
    }
    Ok((values, vectors))
}

/// #2627 — the counted-worst-case relative backward error `η_A` of faer 0.24's
/// real Hessenberg–Schur reduction of an `n × n` matrix.
///
/// Applying `r` Householder or Givens transformations, each of which updates an
/// entry by an inner product over at most `n` terms, one product and one
/// subtraction, perturbs the matrix by at most `r·γ_{n+2}` relative to its
/// Frobenius norm (Higham, *Accuracy and Stability of Numerical Algorithms*, 2nd
/// ed., §19.3). The reduction applies at most:
/// - `2(n − 2)` reflectors for the Hessenberg form, one from each side;
/// - `2·itmax·(n − 1)` bulge-chasing transformations, under faer's iteration cap
///   `itmax = 30·max(10, n)` (`real_schur::lahqr` and `real_schur::multishift_qr`).
///
/// The count takes the iteration cap, not the iterations run, so the band is
/// loose by that ratio.
fn general_eigen_reduction_band(n: usize) -> f64 {
    let itmax = 30 * n.max(10);
    let applications = 2 * n.saturating_sub(2) + 2 * itmax * n.saturating_sub(1);
    applications as f64 * crate::roundoff::accumulation_growth(n + 2)
}

/// `(‖A‖_F², tr A, Σ|a_ii|)`.
fn frobenius_trace_and_diagonal_mass(a: MatRef<'_, f64>) -> (f64, f64, f64) {
    let n = a.nrows();
    let (mut frobenius_sq, mut trace, mut diagonal_mass) = (0.0_f64, 0.0_f64, 0.0_f64);
    for j in 0..n {
        trace += a[(j, j)];
        diagonal_mass += a[(j, j)].abs();
        for i in 0..n {
            frobenius_sq += a[(i, j)] * a[(i, j)];
        }
    }
    (frobenius_sq, trace, diagonal_mass)
}

/// The measured backward error `‖Av − λv‖₂ / ((‖A‖_F + |λ|)·‖v‖₂)` of every
/// slot's eigenpair, with a conjugate pair's two slots carrying the same value.
/// It bounds each eigenvalue's backward error from above. An exact zero residual
/// measures zero, whatever the denominator. `vectors` holds faer's right
/// eigenvectors: a real eigenvalue's in its own column, and for
/// `λ = re[i] + i·im[i]` of a conjugate pair, `col(i) + i·col(i + 1)`.
///
/// `Err((slot, gap))` when the pair starting at `slot` is not adjacent with equal
/// real parts and negated imaginary parts. `gap` is `|re[i+1] − re[i]| + |im[i+1] + im[i]|`,
/// or `|im[i]|` when the pair has no second slot.
fn general_eigenpair_backward_errors(
    a: MatRef<'_, f64>,
    re: &Array1<f64>,
    im: &Array1<f64>,
    vectors: MatRef<'_, f64>,
) -> Result<Array1<f64>, (usize, f64)> {
    let n = a.nrows();
    let (frobenius_sq, _, _) = frobenius_trace_and_diagonal_mass(a);
    let frobenius = frobenius_sq.sqrt();
    let mut av = Mat::<f64>::zeros(n, n);
    faer::linalg::matmul::matmul(av.as_mut(), faer::Accum::Replace, a, vectors, 1.0, pool_parallelism());
    let column_sq = |j: usize| (0..n).map(|k| vectors[(k, j)] * vectors[(k, j)]).sum::<f64>();
    let relative = |residual: f64, scale: f64| if residual == 0.0 { 0.0 } else { residual / scale };
    let mut errors = Array1::<f64>::zeros(n);
    let mut i = 0;
    while i < n {
        if im[i] == 0.0 {
            let lambda = re[i];
            let residual_sq: f64 = (0..n).map(|k| (av[(k, i)] - lambda * vectors[(k, i)]).powi(2)).sum();
            errors[i] = relative(residual_sq.sqrt(), (frobenius + lambda.abs()) * column_sq(i).sqrt());
            i += 1;
        } else {
            if i + 1 >= n {
                return Err((i, im[i].abs()));
            }
            if re[i + 1] != re[i] || im[i + 1] != -im[i] {
                return Err((i, (re[i + 1] - re[i]).abs() + (im[i + 1] + im[i]).abs()));
            }
            // A(x + iy) = (a + ib)(x + iy) is Ax = a·x − b·y and Ay = b·x + a·y.
            let (real, imag) = (re[i], im[i]);
            let residual_sq: f64 = (0..n)
                .map(|k| {
                    let (x, y) = (vectors[(k, i)], vectors[(k, i + 1)]);
                    (av[(k, i)] - real * x + imag * y).powi(2) + (av[(k, i + 1)] - imag * x - real * y).powi(2)
                })
                .sum();
            let vector_norm = (column_sq(i) + column_sq(i + 1)).sqrt();
            let error = relative(residual_sq.sqrt(), (frobenius + real.hypot(imag)) * vector_norm);
            errors[i] = error;
            errors[i + 1] = error;
            i += 2;
        }
    }
    Ok(errors)
}

/// The exact backward error of the eigenvalue `λ = re + i·im` of `A`, as
/// `(σ_min(A − λI) / (‖A‖_F + |λ|), band)`, singular values only, at
/// [`decomposition_parallelism`].
///
/// `σ_min(A − λI)` is the norm of the smallest `E` for which `λ` is an eigenvalue
/// of `A + E`, so it does not depend on how well an eigenvector is determined.
/// faer's eigenvectors are not evidence at a semisimple repeated eigenvalue. Its
/// shifted 2×2 quasi-triangular solve computes `adj(M)·r / det(M)` with
/// `M = B₁ − λI` singular, so the second copy's vector is wrong by an O(1) factor
/// while the eigenvalue is exact (#2627/#2951, probe job 1209094).
///
/// The band charges three things:
/// - the reduction's backward error: `λ` is an eigenvalue of `A + E` with
///   `‖E‖₂ ≤ η_A·‖A‖_F` ([`general_eigen_reduction_band`]);
/// - the SVD's own backward stability ([`crate::roundoff::factor_singular_band`]);
/// - forming `A − λI`, one rounded subtraction per diagonal entry.
fn general_eigenvalue_singular_backward_error(
    a: MatRef<'_, f64>,
    re: f64,
    im: f64,
    reduction: f64,
) -> Result<(f64, f64), svd::SvdError> {
    let n = a.nrows();
    let par = decomposition_parallelism();
    let (frobenius_sq, _, _) = frobenius_trace_and_diagonal_mass(a);
    let magnitude = re.hypot(im);
    let scale = frobenius_sq.sqrt() + magnitude;
    let diagonal_max = (0..n).fold(0.0_f64, |largest, j| largest.max(a[(j, j)].abs()));
    let (sigma_min, sigma_max) = if im == 0.0 {
        let mut shifted = a.to_owned();
        for j in 0..n {
            shifted[(j, j)] -= re;
        }
        let mut s = Diag::<f64>::zeros(n);
        let mut mem = MemBuffer::new(svd::svd_scratch::<f64>(
            n,
            n,
            ComputeSvdVectors::No,
            ComputeSvdVectors::No,
            par,
            Default::default(),
        ));
        svd::svd(shifted.as_ref(), s.as_mut(), None, None, par, MemStack::new(&mut mem), Default::default())?;
        (s.as_ref().column_vector()[n - 1], s.as_ref().column_vector()[0])
    } else {
        let lambda = faer::c64::new(re, im);
        let shifted = Mat::<faer::c64>::from_fn(n, n, |i, j| {
            let value = faer::c64::new(a[(i, j)], 0.0);
            if i == j { value - lambda } else { value }
        });
        let mut s = Diag::<faer::c64>::zeros(n);
        let mut mem = MemBuffer::new(svd::svd_scratch::<faer::c64>(
            n,
            n,
            ComputeSvdVectors::No,
            ComputeSvdVectors::No,
            par,
            Default::default(),
        ));
        svd::svd(shifted.as_ref(), s.as_mut(), None, None, par, MemStack::new(&mut mem), Default::default())?;
        (s.as_ref().column_vector()[n - 1].re, s.as_ref().column_vector()[0].re)
    };
    if scale == 0.0 {
        return Ok((0.0, 0.0));
    }
    let absolute_band = reduction * frobenius_sq.sqrt()
        + crate::roundoff::factor_singular_band(n, n, sigma_max)
        + crate::roundoff::UNIT_ROUNDOFF * (diagonal_max + magnitude);
    let relative = if sigma_min == 0.0 { 0.0 } else { sigma_min / scale };
    Ok((relative, absolute_band / scale))
}

/// Certificate (i)'s band, `η = η_A + 2·γ_{n+1}`, over the reduction's counted
/// backward error `reduction` ([`general_eigen_reduction_band`]). The two
/// `γ_{n+1}` terms charge the eigenvectors' back-substitution and the residual's
/// formation.
fn general_eigen_pair_band(n: usize, reduction: f64) -> f64 {
    reduction + 2.0 * crate::roundoff::accumulation_growth(n + 1)
}

/// Whether certificate (ii)'s trace arm holds ([`general_spectrum_trace_measure`]).
fn general_spectrum_trace_consistent(a: MatRef<'_, f64>, re: &Array1<f64>, reduction: f64) -> bool {
    let (gap, band) = general_spectrum_trace_measure(a, re, reduction);
    gap <= band
}

/// Certificate (ii), the trace, of [`real_general_spectrum`], as
/// `(|Σ re − tr A|, band)`. The computed spectrum is exactly that of `A + E` with
/// `‖E‖_F ≤ η_A·‖A‖_F`, so `Σλ = tr A + tr E` with `|tr E| ≤ √n·‖E‖_F`. Both sums
/// are charged their accumulation bands.
fn general_spectrum_trace_measure(a: MatRef<'_, f64>, re: &Array1<f64>, reduction: f64) -> (f64, f64) {
    let n = a.nrows();
    let (frobenius_sq, trace, diagonal_mass) = frobenius_trace_and_diagonal_mass(a);
    let sum_re: f64 = re.iter().sum();
    let re_mass: f64 = re.iter().map(|value| value.abs()).sum();
    let band = (n as f64).sqrt() * reduction * frobenius_sq.sqrt()
        + crate::roundoff::accumulation_growth(n.saturating_sub(1)) * (diagonal_mass + re_mass);
    ((sum_re - trace).abs(), band)
}

/// Whether certificate (ii)'s Schur arm holds ([`general_spectrum_schur_measure`]).
fn general_spectrum_within_schur_bound(
    a: MatRef<'_, f64>,
    re: &Array1<f64>,
    im: &Array1<f64>,
    reduction: f64,
) -> bool {
    let (magnitude_sq, bound) = general_spectrum_schur_measure(a, re, im, reduction);
    magnitude_sq <= bound
}

/// Certificate (ii), Schur's inequality, of [`real_general_spectrum`], as
/// `(Σ|λ|², bound)`: `Σ|λ|² ≤ ‖A + E‖_F² ≤ (1 + η_A)²·‖A‖_F²`. `‖A‖_F²` is charged
/// the rounding of its `n²` squares and `Σ|λ|²` that of its `2n`. An unconverged
/// slot that inflates a magnitude breaks it.
fn general_spectrum_schur_measure(
    a: MatRef<'_, f64>,
    re: &Array1<f64>,
    im: &Array1<f64>,
    reduction: f64,
) -> (f64, f64) {
    use crate::roundoff::accumulation_growth;
    let n = a.nrows();
    let (frobenius_sq, _, _) = frobenius_trace_and_diagonal_mass(a);
    let magnitude_sq: f64 = re.iter().zip(im.iter()).map(|(r, j)| r * r + j * j).sum();
    let bound = (1.0 + reduction).powi(2) * frobenius_sq * (1.0 + accumulation_growth(n * n))
        + accumulation_growth(2 * n) * magnitude_sq;
    (magnitude_sq, bound)
}

/// #2627 — which measurement certified a slot's eigenvalue in
/// [`real_general_spectrum`].
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum EigenvalueBackwardErrorSource {
    /// The residual `‖Av − λv‖` of faer's right eigenvector, an upper bound on the
    /// eigenvalue's backward error.
    VectorResidual,
    /// The exact backward error `σ_min(A − λI)`. It is measured only where faer's
    /// eigenvector does not certify the eigenvalue, e.g. at a semisimple repeat.
    SingularValue,
}

/// #2627 — a real square matrix's certified spectrum ([`real_general_spectrum`])
/// with the measurements that certified it.
#[derive(Clone, Debug)]
pub struct CertifiedGeneralSpectrum {
    /// Real parts. Conjugate pairs are adjacent, the member with the positive
    /// imaginary part first.
    pub re: Array1<f64>,
    /// Imaginary parts, `0.0` exactly for a real eigenvalue.
    pub im: Array1<f64>,
    /// Per slot, the measured backward error of its eigenvalue relative to
    /// `‖A‖_F + |λ|`, from the source in `sources`. A conjugate pair's two slots carry
    /// the same value.
    pub backward_errors: Array1<f64>,
    /// Per slot, the band its backward error was certified against.
    /// - A `VectorResidual` slot's band is `η = η_A + 2·γ_{n+1}`
    ///   ([`general_eigen_pair_band`]).
    /// - A `SingularValue` slot's band is `η_A` plus the SVD's and the shift's rounding
    ///   ([`general_eigenvalue_singular_backward_error`]).
    ///
    /// `η_A` counts faer's iteration cap, not the iterations run, so it grows like
    /// `60·n³·u`. A decision that needs tighter evidence reads `backward_errors`.
    pub bands: Array1<f64>,
    /// Per slot, which measurement certified it.
    pub sources: Vec<EigenvalueBackwardErrorSource>,
}

/// #2627 — the eigenvalues of a real square matrix at [`evd_parallelism`],
/// certified after faer returns, with the backward errors that certified them.
///
/// Conjugate pairs are adjacent, the member with the positive imaginary part
/// first, and a real eigenvalue has `im == 0.0` exactly. Non-finite input is
/// refused before faer sees it.
///
/// **Why it certifies.** faer 0.24's `evd_real` discards the info return of its
/// Schur QR iteration (`linalg::evd::evd_imp`, mod.rs:1083-1094 in 0.24.0 and
/// 0.24.4). An iteration that exhausts its cap leaves shift estimates in the
/// undeflated slots and still returns `Ok(())`, and the iteration itself is
/// `pub(crate)` in faer. So the helper computes the right eigenvectors internally.
/// It refuses as `GeneralEigenCertificateRefused`, naming the arm, the slot, the
/// measured value and the band, unless all of these hold:
/// - (i) every eigenvalue's backward error is within its band. First the
///   residual of faer's eigenvector ([`general_eigenpair_backward_errors`]) is
///   compared with `η` ([`general_eigen_pair_band`]); it bounds the backward error
///   from above. Where it does not certify, faer's vector is not evidence either
///   way, so the exact backward error `σ_min(A − λI)` is measured instead
///   ([`general_eigenvalue_singular_backward_error`]), one singular-values-only SVD
///   per such eigenvalue.
/// - (ii) `Σ re` agrees with `tr A` ([`general_spectrum_trace_measure`]), and
///   Schur's inequality `Σ|λ|² ≤ ‖A‖_F²` holds
///   ([`general_spectrum_schur_measure`]), each within its band.
/// - Conjugate pairs are adjacent with equal real parts and negated imaginary parts.
///
/// faer's own failures stay `GeneralEigen(EvdError)` and `SvdNoConvergence`.
///
/// An undeflated slot has an O(1) relative backward error, far outside the band.
///
/// **Semisimple repeats.** At a repeated eigenvalue with a full eigenspace, faer's
/// second eigenvector is wrong by an O(1) factor while the eigenvalue is exact, so
/// that slot is certified by `σ_min`.
///
/// **What it does not refuse.** Eigenvalues of a defective or nearly defective
/// matrix, such as a Jordan block with or without a 1e-12 perturbation, are
/// backward-exact: each is an exact eigenvalue of a matrix within the band of `A`.
/// They certify, though their forward error can be far larger than the band. A
/// consumer that needs forward accuracy needs a condition estimate, which this
/// certificate is not.
///
/// **Known limit.** A multiplicity error with a correct eigenvalue sum passes
/// every arm: two slots on one eigenvalue while another is missing.
///
/// **faer dependency.** This reads faer's info-discarding `evd_imp`, its
/// iteration cap, and its conjugate-pair eigenvector convention. Like
/// [`EVD_DEPLOYMENT_DEGREE`], re-read them whenever faer is bumped.
pub fn real_general_spectrum<S: Data<Elem = f64>>(
    matrix: &ArrayBase<S, Ix2>,
) -> Result<CertifiedGeneralSpectrum, FaerLinalgError> {
    let owned = matrix.to_owned();
    let n = owned.nrows();
    if n != owned.ncols() {
        return Err(FaerLinalgError::FactorizationFailed {
            context: "general eigenvalues: non-square input",
        });
    }
    if n == 0 {
        return Ok(CertifiedGeneralSpectrum {
            re: Array1::zeros(0),
            im: Array1::zeros(0),
            backward_errors: Array1::zeros(0),
            bands: Array1::zeros(0),
            sources: Vec::new(),
        });
    }
    if owned.iter().any(|value| !value.is_finite()) {
        return Err(FaerLinalgError::FactorizationFailed {
            context: "general eigenvalues: non-finite input",
        });
    }
    // A 1×1 matrix is its own eigenvalue, exactly; there is nothing to reduce.
    if n == 1 {
        return Ok(CertifiedGeneralSpectrum {
            re: Array1::from_elem(1, owned[[0, 0]]),
            im: Array1::zeros(1),
            backward_errors: Array1::zeros(1),
            bands: Array1::from_elem(1, general_eigen_pair_band(1, general_eigen_reduction_band(1))),
            sources: vec![EigenvalueBackwardErrorSource::VectorResidual],
        });
    }
    let view = FaerArrayView::new(&owned);
    let (re, im, vectors) = catch_unwind(AssertUnwindSafe(|| general_evd(view.as_ref())))
        .map_err(|_| FaerLinalgError::FactorizationFailed {
            context: "general eigendecomposition panic boundary",
        })?
        .map_err(FaerLinalgError::GeneralEigen)?;
    let reduction = general_eigen_reduction_band(n);
    let band = general_eigen_pair_band(n, reduction);
    let mut backward_errors = general_eigenpair_backward_errors(view.as_ref(), &re, &im, vectors.as_ref())
        .map_err(|(slot, gap)| FaerLinalgError::GeneralEigenCertificateRefused {
            arm: "conjugate pairing",
            slot,
            measured: gap,
            band: 0.0,
        })?;
    let mut bands = Array1::from_elem(n, band);
    let mut sources = vec![EigenvalueBackwardErrorSource::VectorResidual; n];
    // One SVD per distinct failing eigenvalue. For a real A, σ_min(A − λI) = σ_min(A − λ̄I), so a value and its
    // conjugate share one measurement, keyed on (re, |im|).
    let mut measured_shifts: Vec<((f64, f64), (f64, f64))> = Vec::new();
    let mut i = 0;
    while i < n {
        let width = if im[i] == 0.0 { 1 } else { 2 };
        if !(backward_errors[i] <= band) {
            let key = (re[i], im[i].abs());
            let (measured, singular_band) = match measured_shifts.iter().find(|(shift, _)| *shift == key) {
                Some(&(_, outcome)) => outcome,
                None => {
                    let outcome = catch_unwind(AssertUnwindSafe(|| {
                        general_eigenvalue_singular_backward_error(view.as_ref(), re[i], im[i], reduction)
                    }))
                    .map_err(|_| FaerLinalgError::FactorizationFailed {
                        context: "general eigenvalue backward-error SVD panic boundary",
                    })?
                    .map_err(|_| FaerLinalgError::SvdNoConvergence {
                        context: "general eigenvalue backward error sigma_min(A - lambda I)",
                    })?;
                    measured_shifts.push((key, outcome));
                    outcome
                }
            };
            if !(measured <= singular_band) {
                return Err(FaerLinalgError::GeneralEigenCertificateRefused {
                    arm: "eigenvalue backward error sigma_min(A - lambda I)",
                    slot: i,
                    measured,
                    band: singular_band,
                });
            }
            for k in i..i + width {
                backward_errors[k] = measured;
                bands[k] = singular_band;
                sources[k] = EigenvalueBackwardErrorSource::SingularValue;
            }
        }
        i += width;
    }
    if !general_spectrum_trace_consistent(view.as_ref(), &re, reduction) {
        let (gap, trace_band) = general_spectrum_trace_measure(view.as_ref(), &re, reduction);
        return Err(FaerLinalgError::GeneralEigenCertificateRefused {
            arm: "trace",
            slot: 0,
            measured: gap,
            band: trace_band,
        });
    }
    if !general_spectrum_within_schur_bound(view.as_ref(), &re, &im, reduction) {
        let (magnitude_sq, schur_bound) = general_spectrum_schur_measure(view.as_ref(), &re, &im, reduction);
        return Err(FaerLinalgError::GeneralEigenCertificateRefused {
            arm: "Schur inequality",
            slot: 0,
            measured: magnitude_sq,
            band: schur_bound,
        });
    }
    let mut im = im;
    let mut i = 0;
    while i < n {
        if im[i] < 0.0 {
            im[i] = -im[i];
            im[i + 1] = -im[i + 1];
        }
        i += if im[i] == 0.0 { 1 } else { 2 };
    }
    Ok(CertifiedGeneralSpectrum { re, im, backward_errors, bands, sources })
}

/// #2627 — the eigenvalues `(re, im)` of [`real_general_spectrum`], for callers
/// that read the spectrum alone.
pub fn real_general_eigenvalues<S: Data<Elem = f64>>(
    matrix: &ArrayBase<S, Ix2>,
) -> Result<(Array1<f64>, Array1<f64>), FaerLinalgError> {
    real_general_spectrum(matrix).map(|spectrum| (spectrum.re, spectrum.im))
}

/// faer's real eigendecomposition with right eigenvectors, at [`evd_parallelism`].
fn general_evd(a: MatRef<'_, f64>) -> Result<(Array1<f64>, Array1<f64>, Mat<f64>), solvers::EvdError> {
    let n = a.nrows();
    let par = evd_parallelism();
    let mut re = Diag::<f64>::zeros(n);
    let mut im = Diag::<f64>::zeros(n);
    let mut vectors = Mat::<f64>::zeros(n, n);
    let mut mem = MemBuffer::new(faer::linalg::evd::evd_scratch::<f64>(
        n,
        faer::linalg::evd::ComputeEigenvectors::No,
        faer::linalg::evd::ComputeEigenvectors::Yes,
        par,
        Default::default(),
    ));
    faer::linalg::evd::evd_real(
        a,
        re.as_mut(),
        im.as_mut(),
        None,
        Some(vectors.as_mut()),
        par,
        MemStack::new(&mut mem),
        Default::default(),
    )?;
    Ok((diag_to_array(re.as_ref()), diag_to_array(im.as_ref()), vectors))
}

impl<S: Data<Elem = f64>> FaerEigh for ArrayBase<S, Ix2> {
    fn eigh(&self, side: Side) -> Result<(Array1<f64>, Array2<f64>), FaerLinalgError> {
        fn try_eigh(
            matrix: &Array2<f64>,
            side: Side,
        ) -> Result<(Array1<f64>, Array2<f64>), FaerLinalgError> {
            let faerview = FaerArrayView::new(matrix);
            // #2267/#2738 — time the decomposition and name the parallelism that
            // governed it: the degree passed to faer per call (#2627), never
            // faer's process-global policy.
            //
            // This is `O(dim^3)`, so at `dim` in the thousands the difference
            // between sequential and a wide pool is hours. #2267 lost two
            // three-hour jobs to a decomposition that was silent about both its
            // duration and its parallelism; the count makes "one slow call or
            // many?" answerable without a second run.
            let eigh_started = std::time::Instant::now();
            let eigh_par = evd_parallelism();
            let (s, u) = catch_unwind(AssertUnwindSafe(|| self_adjoint_evd(faerview.as_ref(), side)))
                .map_err(|_| FaerLinalgError::FactorizationFailed {
                    context: "self-adjoint eigendecomposition panic boundary",
                })?
                .map_err(FaerLinalgError::SelfAdjointEigen)?;
            let eigh_elapsed = eigh_started.elapsed();
            let eigh_calls = EIGH_CALLS.fetch_add(1, Ordering::Relaxed) + 1;
            let eigh_nanos_total = EIGH_NANOS
                .fetch_add(eigh_elapsed.as_nanos() as u64, Ordering::Relaxed)
                + eigh_elapsed.as_nanos() as u64;
            log::debug!(
                "[eigh] dim={} elapsed={:.3}s faer_parallelism={:?} \
                 calls_so_far={eigh_calls} cumulative={:.3}s",
                matrix.nrows(),
                eigh_elapsed.as_secs_f64(),
                eigh_par,
                eigh_nanos_total as f64 / 1e9,
            );
            let values = diag_to_array(s.as_ref());
            let vectors = mat_to_array(u.as_ref());
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
            .fold(0.0_f64, |acc, &value| acc.max(value.abs()));
        if scale == 0.0 {
            // The zero matrix has nothing to scale by; its eigenpairs are exactly
            // (0, I).
            let n = repaired.nrows();
            return Ok((Array1::zeros(n), Array2::eye(n)));
        }
        let scaled = repaired.mapv(|value| value / scale);
        // The repair is exact preconditioning and nothing else: the symmetrized
        // matrix divided by its own magnitude, decomposed once. A decomposition
        // that still fails or returns non-finite pairs is refused. It is never
        // retried on a diagonally shifted matrix until some shift happens to
        // succeed (SPEC rule 21, #2902).
        let (mut evals, evecs) = try_eigh(&scaled, side)?;
        if evals.iter().any(|value| !value.is_finite())
            || evecs.iter().any(|value| !value.is_finite())
        {
            return Err(FaerLinalgError::SelfAdjointEigenNonFiniteInput {
                context: "self-adjoint eigendecomposition repaired output validation",
            });
        }
        for value in &mut evals {
            *value *= scale;
        }
        Ok((evals, evecs))
    }
}

pub struct FaerCholeskyFactor {
    factor: FaerLlt<f64>,
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

    pub(crate) fn solve_mat_into<S: Data<Elem = f64>>(
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
        diag_to_array(self.factor.lower().diagonal())
    }

    pub fn lower_triangular(&self) -> Array2<f64> {
        mat_to_array(self.factor.lower())
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
        cholesky_factor_logdet(self.factor.lower())
    }
}

pub trait FaerCholesky {
    fn cholesky(&self, side: Side) -> Result<FaerCholeskyFactor, FaerLinalgError>;
}

impl<S: Data<Elem = f64>> FaerCholesky for ArrayBase<S, Ix2> {
    fn cholesky(&self, side: Side) -> Result<FaerCholeskyFactor, FaerLinalgError> {
        let faerview = FaerArrayView::new(self);
        let factor = FaerLlt::new(faerview.as_ref(), side).map_err(FaerLinalgError::Cholesky)?;
        Ok(FaerCholeskyFactor { factor })
    }
}

pub trait FaerQr {
    fn qr(&self) -> Result<(Array2<f64>, Array2<f64>), FaerLinalgError>;
}

impl<S: Data<Elem = f64>> FaerQr for ArrayBase<S, Ix2> {
    fn qr(&self) -> Result<(Array2<f64>, Array2<f64>), FaerLinalgError> {
        let faerview = FaerArrayView::new(self);
        let (basis, coeff, r) = householder_qr(faerview.as_ref());
        let mut q = Mat::<f64>::identity(basis.nrows(), basis.ncols());
        apply_householder_on_the_left(basis.as_ref(), coeff.as_ref(), q.as_mut());
        Ok((mat_to_array(q.as_ref()), mat_to_array(r.as_ref())))
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
    let qr = ColumnPivotedQr::new(faerview.as_ref());
    let r = qr.thin_r();
    let diag_len = r.nrows().min(r.ncols());
    let leading_diag = if diag_len > 0 { r[(0, 0)].abs() } else { 0.0 };
    let tol = match cutoff {
        RrqrRankCutoff::RelativeAlpha(rank_alpha) => {
            rank_alpha * f64::EPSILON * (a.nrows().max(a.ncols()).max(1) as f64) * leading_diag
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
        apply_householder_on_the_left(qr.basis.as_ref(), qr.coeff.as_ref(), selector.as_mut());
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
    let qr = ColumnPivotedQr::new(faerview.as_ref());
    let r = qr.thin_r();
    let diag_len = r.nrows().min(r.ncols());
    let leading_diag = if diag_len > 0 { r[(0, 0)].abs() } else { 0.0 };
    // The pivoted QR's backward error is `α·ε·max(m, p)·|R₁₁|`, in the matrix's own
    // units: a pivot inside it is zero to the precision the factorization has.
    let tol = rank_alpha * f64::EPSILON * (a.nrows().max(a.ncols()).max(1) as f64) * leading_diag;
    let rank = (0..diag_len).filter(|&i| r[(i, i)].abs() > tol).count();
    let column_permutation = qr.forward.clone();
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
    let qr = ColumnPivotedQr::new(faer_f.as_ref());
    let r = qr.thin_r();
    let diag_len = r.nrows().min(r.ncols());
    let pivots: Vec<f64> = (0..diag_len).map(|i| r[(i, i)].abs()).collect();
    let leading_diag = pivots.first().copied().unwrap_or(0.0);
    let column_permutation = qr.forward.clone();
    // Re-scale the tolerance from F's `max(p, p)=p` row dimension to the
    // original tall design's `max(m_rows, p)`, keeping the rank cut bit-
    // identical to what the tall [`rrqr_with_permutation`] would have produced.
    let tol = rank_alpha * f64::EPSILON * (m_rows.max(p).max(1) as f64) * leading_diag;
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
    // Exactly zero dropped pivots sit infinitely far below any cutoff.
    let dropped_margin = if rank == diag_len || max_dropped == 0.0 {
        f64::INFINITY
    } else {
        tol / max_dropped
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
    // A kept pivot implies a positive leading pivot, so this floor is positive
    // wherever it divides.
    let gram_precision_floor = f64::EPSILON.sqrt() * leading_diag;
    let kept_floor_margin = if rank == 0 {
        f64::INFINITY
    } else {
        min_kept / gram_precision_floor
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

    /// The `fma,avx2` variants of every dispatched kernel are bit-identical
    /// to the baseline bodies: the dispatch changes how an FMA is executed,
    /// never what it computes. Run on the same ill-conditioned ensemble the
    /// accuracy gate uses, so any lane reassociation would surface as a
    /// changed bit. The CPU must execute the variants for this to be a
    /// comparison at all, so a machine without `fma,avx2` fails loudly
    /// instead of comparing a body with itself.
    #[cfg(target_arch = "x86_64")]
    #[test]
    fn fma_avx2_kernel_variants_are_bit_identical_to_the_baseline_bodies() {
        assert!(
            super::fma_avx2_available(),
            "this machine reports no fma/avx2: the variant path cannot be exercised here"
        );
        for seed in 0..64u64 {
            let len = 200 + (seed as usize % 57);
            let (a, b) = ill_conditioned_pair(len, 0x9E37_79B9 ^ seed.wrapping_mul(2654435761));
            // SAFETY: the assertion above established the CPU features these
            // variants enable.
            let (dot_v, std_v) = unsafe {
                (
                    super::fma_dot_fma_avx2(&a, &b),
                    super::standard_fma_dot_fma_avx2(&a, &b),
                )
            };
            assert_eq!(dot_v.to_bits(), super::fma_dot_body(&a, &b).to_bits(), "fma_dot seed={seed}");
            assert_eq!(
                std_v.to_bits(),
                super::standard_fma_dot_body(&a, &b).to_bits(),
                "standard_fma_dot seed={seed}"
            );
            // Xᵀv block partial: `len` rows of width 7 (a remainder-bearing
            // width), and the axpy over the same data.
            let p = 7;
            let rows: Vec<f64> = (0..len * p).map(|k| a[k % len] * (1.0 + (k % 3) as f64)).collect();
            let mut acc_body = vec![0.0f64; p];
            let mut acc_var = vec![0.0f64; p];
            super::atv_block_accumulate_body(&rows, &b, &mut acc_body);
            // SAFETY: as above.
            unsafe { super::atv_block_accumulate_fma_avx2(&rows, &b, &mut acc_var) };
            assert_eq!(
                acc_var.iter().map(|v| v.to_bits()).collect::<Vec<_>>(),
                acc_body.iter().map(|v| v.to_bits()).collect::<Vec<_>>(),
                "atv block seed={seed}"
            );
            let mut y_body = b.clone();
            let mut y_var = b.clone();
            super::fma_axpy_into_body(a[0], &a, &mut y_body);
            // SAFETY: as above.
            unsafe { super::fma_axpy_into_fma_avx2(a[0], &a, &mut y_var) };
            assert_eq!(
                y_var.iter().map(|v| v.to_bits()).collect::<Vec<_>>(),
                y_body.iter().map(|v| v.to_bits()).collect::<Vec<_>>(),
                "axpy seed={seed}"
            );
        }
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
mod evd_degree_2627_tests {
    use super::*;

    fn words(values: &Diag<f64>, vectors: &Mat<f64>) -> Vec<u64> {
        let mut out: Vec<u64> = values.as_ref().column_vector().iter().map(|v| v.to_bits()).collect();
        for j in 0..vectors.ncols() {
            for i in 0..vectors.nrows() {
                out.push(vectors[(i, j)].to_bits());
            }
        }
        out
    }

    fn sequential_words(a: MatRef<'_, f64>) -> Vec<u64> {
        let n = a.nrows();
        let mut s = Diag::<f64>::zeros(n);
        let mut u = Mat::<f64>::zeros(n, n);
        let mut mem = MemBuffer::new(faer::linalg::evd::self_adjoint_evd_scratch::<f64>(
            n,
            faer::linalg::evd::ComputeEigenvectors::Yes,
            Par::Seq,
            Default::default(),
        ));
        faer::linalg::evd::self_adjoint_evd(
            a,
            s.as_mut(),
            Some(u.as_mut()),
            Par::Seq,
            MemStack::new(&mut mem),
            Default::default(),
        )
        .expect("sequential EVD");
        words(&s, &u)
    }

    /// Every self-adjoint EVD carries the same words on pools of width 1, 4 and
    /// 24, because its degree is [`evd_parallelism`] rather than the pool's.
    /// The positive control shows that at this dimension a different degree
    /// gives different words, so the pin cannot pass below faer's parallel
    /// threshold.
    #[test]
    fn self_adjoint_evd_words_do_not_depend_on_the_pool_width_2627() {
        let n = 1500usize;
        let mut state = 0x2627_E7D0_u64;
        let mut a = Mat::<f64>::zeros(n, n);
        for j in 0..n {
            for i in j..n {
                state = state
                    .wrapping_mul(6_364_136_223_846_793_005)
                    .wrapping_add(1_442_695_040_888_963_407);
                let value = (state >> 11) as f64 / (1_u64 << 53) as f64 - 0.5;
                a[(i, j)] = value;
                a[(j, i)] = value;
            }
        }
        let at_width = |width: usize| {
            let pool = rayon::ThreadPoolBuilder::new().num_threads(width).build().expect("pool");
            pool.install(|| {
                let (values, vectors) = self_adjoint_evd(a.as_ref(), Side::Lower).expect("EVD");
                words(&values, &vectors)
            })
        };
        let single = at_width(1);
        assert_eq!(single, at_width(4));
        assert_eq!(single, at_width(24));
        assert_ne!(
            single,
            sequential_words(a.as_ref()),
            "at n={n} the degree must change the words, or this pin is vacuous"
        );
    }

    /// The deployment degree reaches faer as the chunk count of its tridiagonal
    /// reduction. At n = 1500, above faer's parallel grain, a neighbouring degree
    /// partitions the trailing matvec differently and gives different words (degree
    /// 22 against 24 in job 1180582). If a faer change stopped reading the degree,
    /// or capped its split count below it, these words would agree, and the words
    /// pin above could keep passing with the deployment degree inert.
    #[test]
    fn evd_deployment_degree_reaches_faer_as_its_chunk_count_2627() {
        assert!(
            matches!(evd_parallelism(), Par::Rayon(degree) if degree.get() == EVD_DEPLOYMENT_DEGREE),
            "the self-adjoint EVD must hand faer the deployment degree"
        );
        let n = 1500usize;
        let mut state = 0x2627_E7D0_u64;
        let mut a = Mat::<f64>::zeros(n, n);
        for j in 0..n {
            for i in j..n {
                state = state
                    .wrapping_mul(6_364_136_223_846_793_005)
                    .wrapping_add(1_442_695_040_888_963_407);
                let value = (state >> 11) as f64 / (1_u64 << 53) as f64 - 0.5;
                a[(i, j)] = value;
                a[(j, i)] = value;
            }
        }
        let words_at = |par: Par| {
            let mut s = Diag::<f64>::zeros(n);
            let mut u = Mat::<f64>::zeros(n, n);
            let mut mem = MemBuffer::new(faer::linalg::evd::self_adjoint_evd_scratch::<f64>(
                n,
                faer::linalg::evd::ComputeEigenvectors::Yes,
                par,
                Default::default(),
            ));
            faer::linalg::evd::self_adjoint_evd(
                a.as_ref(),
                s.as_mut(),
                Some(u.as_mut()),
                par,
                MemStack::new(&mut mem),
                Default::default(),
            )
            .expect("EVD at a fixed degree");
            words(&s, &u)
        };
        assert_ne!(
            words_at(Par::rayon(EVD_DEPLOYMENT_DEGREE)),
            words_at(Par::rayon(EVD_DEPLOYMENT_DEGREE - 2)),
            "at n={n} a neighbouring degree must change the words: the degree is faer's chunk count"
        );
    }
}

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

}

#[cfg(test)]
mod eigh_ordering_contract_tests {
    use super::*;
    use ndarray::Array2;

    /// `FaerEigh::eigh` returns eigenvalues in ASCENDING order, and at least one
    /// consumer's correctness depends on it while nothing pinned it.
    ///
    /// `gam-sae`'s `canonicalize_exact_a_rank_clusters`
    /// (`crates/gam-sae/src/manifold/exact_stationarity_krylov.rs`) finds each
    /// numerically repeated cluster with
    ///
    /// ```text
    /// while end < values.len() && values[end] - values[start] <= envelope { end += 1; }
    /// ```
    ///
    /// — a scan for a RUN of equal values, which only enumerates a cluster when
    /// equal eigenvalues are ADJACENT. Adjacency is a consequence of sorting and
    /// of nothing else. If the underlying driver ever returned an unsorted
    /// spectrum, that loop would not error: it would silently see clusters of
    /// width 1 where a cluster exists, skip the within-cluster re-diagonalisation
    /// entirely, and return a basis that is not stable under the perturbation the
    /// function is named for. A silent wrong answer, from a dependency upgrade,
    /// with no test between it and the fit.
    ///
    /// Measured 2026-09-05 while attributing 1,383,210 `eigh` calls in one hung
    /// SAE test: the exact-A spectral block is one of the two callers the native
    /// stacks caught in the act, reached from `terminal_exact_newton_polish` ->
    /// `materialize_exact_stationarity_geometry` -> `exact_hessian_spectral_block`.
    /// The other is `gam_solve::arrow_schur::factorization::row_sub_floor_null_directions`.
    #[test]
    fn eigh_returns_eigenvalues_in_ascending_order() {
        fn hashed_unit(seed: u64) -> f64 {
            let mut z = seed.wrapping_add(0x9E37_79B9_7F4A_7C15);
            z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
            z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
            z ^= z >> 31;
            ((z >> 11) as f64 / (1u64 << 53) as f64) * 2.0 - 1.0
        }

        // Dimensions bracketing the ones the SAE fit actually asks for (1, 2, 3,
        // 5, 6, 9, 36) plus a size where a driver could plausibly switch
        // algorithm, and scales spanning 18 decades so the check is not made on
        // one magnitude.
        for &n in &[1_usize, 2, 3, 5, 6, 9, 17, 36] {
            for seed in 0..8_u64 {
                for &scale in &[1.0_f64, 1.0e-9, 1.0e9] {
                    let mut m = Array2::<f64>::zeros((n, n));
                    let mut k = seed.wrapping_mul(1_000_003).wrapping_add(n as u64);
                    for i in 0..n {
                        for j in 0..=i {
                            k = k.wrapping_add(0x1234_5678);
                            let value = hashed_unit(k) * scale;
                            m[[i, j]] = value;
                            m[[j, i]] = value;
                        }
                    }
                    let (values, _) = m.eigh(Side::Lower).expect("eigendecomposition");
                    assert_eq!(values.len(), n, "n={n}: one eigenvalue per dimension");
                    for w in 1..n {
                        assert!(
                            values[w - 1] <= values[w],
                            "n={n} seed={seed} scale={scale:e}: eigenvalues are NOT ascending at \
                             index {w} ({:e} then {:e}). `canonicalize_exact_a_rank_clusters` scans for RUNS of \
                             equal eigenvalues and would silently stop finding degenerate clusters.",
                            values[w - 1],
                            values[w]
                        );
                    }
                }
            }
        }
    }

    /// The ordering claim above is only load-bearing because equal eigenvalues
    /// land ADJACENT. Assert that directly on a planted degeneracy rather than
    /// inferring it from sortedness, so the property the consumer actually uses
    /// is the property under test.
    #[test]
    fn equal_eigenvalues_are_returned_adjacent() {
        // diag(2, 7, 2, 7, 2) in a rotated basis: three eigenvalues at 2 and two
        // at 7, planted so the degeneracy is exact rather than incidental.
        let d = ndarray::arr1(&[2.0_f64, 7.0, 2.0, 7.0, 2.0]);
        let n = d.len();
        // A Householder reflector Q = I - 2vv^T/(v^Tv) is orthogonal and exact
        // enough here that Q diag(d) Q^T keeps the spectrum to round-off.
        let v = ndarray::arr1(&[1.0_f64, -2.0, 3.0, -4.0, 5.0]);
        let vtv: f64 = v.iter().map(|x| x * x).sum();
        let mut q = Array2::<f64>::zeros((n, n));
        for i in 0..n {
            q[[i, i]] = 1.0;
        }
        for i in 0..n {
            for j in 0..n {
                q[[i, j]] -= 2.0 * v[i] * v[j] / vtv;
            }
        }
        let mut a = Array2::<f64>::zeros((n, n));
        for i in 0..n {
            for j in 0..n {
                let mut acc = 0.0;
                for k in 0..n {
                    acc += q[[i, k]] * d[k] * q[[j, k]];
                }
                a[[i, j]] = acc;
            }
        }
        let (values, _) = a.eigh(Side::Lower).expect("eigendecomposition");
        // Three near-2 then two near-7, contiguously. A tolerance is needed
        // because the similarity transform is floating point; the ADJACENCY is
        // what is under test, not the digits.
        let low = values.iter().filter(|v| (**v - 2.0).abs() < 1.0e-9).count();
        let high = values.iter().filter(|v| (**v - 7.0).abs() < 1.0e-9).count();
        assert_eq!(low, 3, "planted multiplicity 3 at lambda=2, got {values:?}");
        assert_eq!(high, 2, "planted multiplicity 2 at lambda=7, got {values:?}");
        for w in 0..3 {
            assert!(
                (values[w] - 2.0).abs() < 1.0e-9,
                "the three lambda=2 eigenvalues must occupy indices 0..3 contiguously, \
                 or `canonicalize_exact_a_rank_clusters`'s run scan splits the cluster: {values:?}"
            );
        }
        for w in 3..5 {
            assert!(
                (values[w] - 7.0).abs() < 1.0e-9,
                "the two lambda=7 eigenvalues must occupy indices 3..5 contiguously: {values:?}"
            );
        }
    }
}

#[cfg(test)]
mod general_eigenvalues_2627_tests {
    use super::*;
    use crate::roundoff::accumulation_growth;

    fn hashed_matrix(n: usize, seed: u64) -> Array2<f64> {
        let mut state = seed.wrapping_mul(0x9E37_79B9_7F4A_7C15).wrapping_add(0x2627_6E1A);
        Array2::from_shape_fn((n, n), |_| {
            state = state
                .wrapping_mul(6_364_136_223_846_793_005)
                .wrapping_add(1_442_695_040_888_963_407);
            (state >> 11) as f64 / (1_u64 << 53) as f64 - 0.5
        })
    }

    fn frobenius(a: &Array2<f64>) -> f64 {
        a.iter().map(|value| value * value).sum::<f64>().sqrt()
    }

    /// A normal matrix with a planted spectrum (two rotation–scaling blocks and
    /// one real eigenvalue) turned by a Householder reflector `Q`. By Bauer–Fike
    /// on the exact normal `QBQᵀ`, every computed eigenvalue is within
    /// `‖E‖₂ + ‖F‖₂` of a planted one. `E` is the reduction's backward error, at
    /// most `η_A·‖A‖_F`. `F` is the rounding of forming `QBQᵀ`, two products of
    /// `n`-term inner products, at most `γ_{2n+2}·‖A‖_F`.
    #[test]
    fn real_general_eigenvalues_recovers_a_planted_rotation_spectrum_2627() {
        let n = 5;
        let (r1, t1, r2, t2, real) = (2.0_f64, 0.7_f64, 0.5_f64, 2.1_f64, -1.25_f64);
        let mut b = Array2::<f64>::zeros((n, n));
        for (offset, r, t) in [(0_usize, r1, t1), (3_usize, r2, t2)] {
            b[[offset, offset]] = r * t.cos();
            b[[offset, offset + 1]] = -r * t.sin();
            b[[offset + 1, offset]] = r * t.sin();
            b[[offset + 1, offset + 1]] = r * t.cos();
        }
        b[[2, 2]] = real;
        let v = ndarray::arr1(&[1.0_f64, -2.0, 3.0, -4.0, 5.0]);
        let vtv = v.dot(&v);
        let mut q = Array2::<f64>::eye(n);
        for i in 0..n {
            for j in 0..n {
                q[[i, j]] -= 2.0 * v[i] * v[j] / vtv;
            }
        }
        let a = q.dot(&b).dot(&q.t());
        let (re, im) = real_general_eigenvalues(&a).expect("certified spectrum");
        let band = (general_eigen_reduction_band(n) + accumulation_growth(2 * n + 2)) * frobenius(&a);
        let mut planted = vec![(r1 * t1.cos(), r1 * t1.sin()), (real, 0.0), (r2 * t2.cos(), r2 * t2.sin())];
        let mut i = 0;
        while i < n {
            let found = (re[i], im[i]);
            let position = planted
                .iter()
                .position(|&(pr, pi)| (pr - found.0).hypot(pi - found.1) <= band)
                .unwrap_or_else(|| panic!("eigenvalue {found:?} is within {band:.3e} of no planted one: {planted:?}"));
            planted.remove(position);
            i += if im[i] == 0.0 { 1 } else { 2 };
        }
        assert!(planted.is_empty(), "planted eigenvalues left unrecovered: {planted:?}");
    }

    /// The ordering contract on hashed matrices. A conjugate pair is adjacent,
    /// with equal real parts, negated imaginary parts and the positive member
    /// first. A real eigenvalue has `im == 0.0` exactly. At least one pair must
    /// appear across the sweep, or the contract was never exercised.
    #[test]
    fn real_general_eigenvalues_orders_conjugate_pairs_positive_first_2627() {
        let mut pairs = 0_usize;
        for &n in &[1_usize, 2, 3, 5, 17, 40] {
            for seed in 0..4_u64 {
                let a = hashed_matrix(n, seed);
                let (re, im) = real_general_eigenvalues(&a).expect("certified spectrum");
                assert_eq!((re.len(), im.len()), (n, n));
                let mut i = 0;
                while i < n {
                    if im[i] == 0.0 {
                        i += 1;
                        continue;
                    }
                    assert!(im[i] > 0.0, "n={n} seed={seed}: the pair at {i} starts with im {}", im[i]);
                    assert!(i + 1 < n, "n={n} seed={seed}: a pair member at the last index");
                    assert_eq!(re[i + 1], re[i], "n={n} seed={seed}: the pair at {i} has unequal real parts");
                    assert_eq!(im[i + 1], -im[i], "n={n} seed={seed}: the pair at {i} is not conjugate");
                    pairs += 1;
                    i += 2;
                }
            }
        }
        assert!(pairs > 0, "no conjugate pair appeared, so the ordering contract was never exercised");
    }

    /// Non-finite input is refused before faer sees it, never as faer's
    /// `NoConvergence`, which would misname the cause.
    #[test]
    fn real_general_eigenvalues_refuses_non_finite_input_before_faer_2627() {
        for poison in [f64::NAN, f64::INFINITY, f64::NEG_INFINITY] {
            let mut a = hashed_matrix(4, 7);
            a[[2, 1]] = poison;
            match real_general_eigenvalues(&a) {
                Err(FaerLinalgError::FactorizationFailed { context }) => {
                    assert!(context.contains("non-finite"), "refused for another reason: {context}")
                }
                other => panic!("{poison} input must be refused before faer: {other:?}"),
            }
        }
    }

    /// Each certificate arm refuses output it cannot vouch for, for its own
    /// reason. faer's converged output on a hashed matrix passes every arm; that
    /// is the negative control. Then:
    /// - arm (i): the leading eigenvalue (both members if it is a pair, so the pair
    ///   stays conjugate) moved by `10⁶·η·(‖A‖_F + |λ|)` fails the backward-error
    ///   arm. An unconverged slot must meet that refusal;
    /// - arm (i): a pair broken off conjugacy fails it;
    /// - arm (ii): `Σ re` moved by `‖A‖_F` fails the trace arm;
    /// - arm (ii): two real parts moved by `±s` with `s = 4‖A‖_F` keep the trace
    ///   and fail Schur's inequality. `(λ_k + s)² + (λ_j − s)² − λ_k² − λ_j²` is at
    ///   least `2s(s − |λ_k − λ_j|)`, which is at least `2s(s − 2‖A‖_F) = 16‖A‖_F²`.
    #[test]
    fn general_eigen_certificate_arms_refuse_corrupted_spectra_2627() {
        let n = 12;
        let a = hashed_matrix(n, 3);
        let view = FaerArrayView::new(&a);
        let (re, im, vectors) = general_evd(view.as_ref()).expect("faer eigendecomposition");
        let reduction = general_eigen_reduction_band(n);
        let eta = general_eigen_pair_band(n, reduction);
        let within_band = |re: &Array1<f64>, im: &Array1<f64>| {
            general_eigenpair_backward_errors(view.as_ref(), re, im, vectors.as_ref())
                .is_ok_and(|errors| errors.iter().all(|error| *error <= eta))
        };
        assert!(within_band(&re, &im));
        assert!(general_spectrum_trace_consistent(view.as_ref(), &re, reduction));
        assert!(general_spectrum_within_schur_bound(view.as_ref(), &re, &im, reduction));
        let scale = frobenius(&a);

        let shift = 1.0e6 * eta * (scale + re[0].hypot(im[0]));
        let mut moved = re.clone();
        moved[0] += shift;
        if im[0] != 0.0 {
            moved[1] += shift;
        }
        assert!(
            !within_band(&moved, &im),
            "an eigenvalue moved by {shift:.3e} must fail the backward-error arm"
        );
        // The fallback must not rescue it: the exact backward error sigma_min(A - lambda I) of the moved
        // eigenvalue is outside its own band too.
        let (measured, singular_band) =
            general_eigenvalue_singular_backward_error(view.as_ref(), moved[0], im[0], reduction).expect("SVD");
        assert!(
            measured > singular_band,
            "a moved eigenvalue must fail sigma_min too: measured {measured:.3e} against band {singular_band:.3e}"
        );

        let mut first_pair = None;
        let mut i = 0;
        while i < n {
            if im[i] == 0.0 {
                i += 1;
            } else {
                first_pair = Some(i);
                break;
            }
        }
        let k = first_pair.expect("the hashed 12×12 matrix has a conjugate pair");
        let mut broken = im.clone();
        broken[k + 1] *= 0.5;
        assert!(
            !within_band(&re, &broken),
            "a pair broken off conjugacy must fail the backward-error arm"
        );

        let mut off_trace = re.clone();
        off_trace[0] += scale;
        assert!(
            !general_spectrum_trace_consistent(view.as_ref(), &off_trace, reduction),
            "a spectrum whose sum is off the trace by ‖A‖_F must fail the trace arm"
        );

        let s = 4.0 * scale;
        let mut inflated = re.clone();
        inflated[0] += s;
        inflated[1] -= s;
        assert!(
            general_spectrum_trace_consistent(view.as_ref(), &inflated, reduction),
            "the inflation keeps the trace, so only Schur's inequality may refuse it"
        );
        assert!(
            !general_spectrum_within_schur_bound(view.as_ref(), &inflated, &im, reduction),
            "real parts moved by ±{s:.3e} must fail Schur's inequality"
        );
    }

    /// The measured backward errors a consumer reads beside the spectrum. Each is
    /// finite and within its slot's band, and a conjugate pair's two slots carry one
    /// measurement, one band and one source. `real_general_eigenvalues` returns
    /// exactly the spectrum's `(re, im)`. These well-separated hashed spectra are
    /// certified by faer's eigenvector residuals alone, with no SVD.
    #[test]
    fn real_general_spectrum_reports_measured_backward_errors_within_its_band_2627() {
        for &n in &[1_usize, 2, 5, 17, 40] {
            let a = hashed_matrix(n, 11);
            let spectrum = real_general_spectrum(&a).expect("certified spectrum");
            assert_eq!((spectrum.backward_errors.len(), spectrum.bands.len(), spectrum.sources.len()), (n, n, n));
            for k in 0..n {
                let (error, band) = (spectrum.backward_errors[k], spectrum.bands[k]);
                assert!(band.is_finite() && band > 0.0, "n={n}: slot {k} band {band}");
                assert!(error.is_finite() && error <= band, "n={n}: slot {k} backward error {error:e} against band {band:e}");
                assert_eq!(spectrum.sources[k], EigenvalueBackwardErrorSource::VectorResidual, "n={n}: slot {k}");
            }
            let mut i = 0;
            while i < n {
                if spectrum.im[i] == 0.0 {
                    i += 1;
                    continue;
                }
                assert_eq!(
                    (spectrum.backward_errors[i], spectrum.bands[i], spectrum.sources[i]),
                    (spectrum.backward_errors[i + 1], spectrum.bands[i + 1], spectrum.sources[i + 1]),
                    "n={n}: the pair at {i} carries two different measurements"
                );
                i += 2;
            }
            let (re, im) = real_general_eigenvalues(&a).expect("certified spectrum");
            assert_eq!(re, spectrum.re, "n={n}: the convenience returns other real parts");
            assert_eq!(im, spectrum.im, "n={n}: the convenience returns other imaginary parts");
        }
    }

    /// mpd-schur's exact-dyadic n = 8 fixture: T = G·B·G⁻¹ with G = I + N (ones on
    /// the first superdiagonal, minus ones on the second) and
    /// B = diag(R(0.5, 0.75), R(0.5, 0.75), 2, −1, R(−0.625, 0.25)),
    /// R(a, b) = [[a, −b], [b, a]]. T·G = G·B holds bitwise.
    fn semisimple_fixture() -> (Array2<f64>, Array2<f64>) {
        let n = 8;
        let mut g = Array2::<f64>::eye(n);
        for i in 0..n {
            if i + 1 < n {
                g[[i, i + 1]] = 1.0;
            }
            if i + 2 < n {
                g[[i, i + 2]] = -1.0;
            }
        }
        let mut g_inverse = Array2::<f64>::zeros((n, n));
        for j in 0..n {
            for i in (0..n).rev() {
                let mut s = if i == j { 1.0 } else { 0.0 };
                for k in i + 1..n {
                    s -= g[[i, k]] * g_inverse[[k, j]];
                }
                g_inverse[[i, j]] = s;
            }
        }
        let mut b = Array2::<f64>::zeros((n, n));
        for (o, re, im) in [(0_usize, 0.5_f64, 0.75_f64), (2, 0.5, 0.75), (6, -0.625, 0.25)] {
            b[[o, o]] = re;
            b[[o, o + 1]] = -im;
            b[[o + 1, o]] = im;
            b[[o + 1, o + 1]] = re;
        }
        b[[4, 4]] = 2.0;
        b[[5, 5]] = -1.0;
        assert_eq!(g.dot(&g_inverse), Array2::<f64>::eye(n), "G⁻¹ must be exact");
        let t = g.dot(&b).dot(&g_inverse);
        assert_eq!(t.dot(&g), g.dot(&b), "T·G = G·B must hold bitwise");
        (t, g)
    }

    /// A semisimple repeated pair certifies (#2627/#2951). faer's eigenvector for the
    /// repeat is wrong by an O(1) factor, so its residual refuses; that is the control
    /// showing the fixture exercises the fallback. The exact backward error
    /// `σ_min(T − λI)` certifies the exact eigenvalue instead. Every returned eigenvalue
    /// is within `κ₂(G)·‖E‖₂` of a planted one (Bauer–Fike for the diagonalizable
    /// `T = (G·W)·D·(G·W)⁻¹`, with `W` the unitary 2×2 block rotations). `‖E‖₂` is the
    /// slot's band times `‖T‖_F + |λ|`.
    #[test]
    fn real_general_spectrum_certifies_a_semisimple_repeated_pair_2627() {
        let (t, g) = semisimple_fixture();
        let n = t.nrows();
        let view = FaerArrayView::new(&t);
        let (re, im, vectors) = general_evd(view.as_ref()).expect("faer eigendecomposition");
        let vector_band = general_eigen_pair_band(n, general_eigen_reduction_band(n));
        let vector_errors = general_eigenpair_backward_errors(view.as_ref(), &re, &im, vectors.as_ref()).expect("pairing");
        assert!(
            vector_errors.iter().any(|error| *error > vector_band),
            "the fixture must defeat faer's eigenvector residual, or this pin does not reach sigma_min: {vector_errors:?}"
        );

        let spectrum = real_general_spectrum(&t).expect("a semisimple repeated pair certifies");
        let singular: Vec<usize> =
            (0..n).filter(|&k| spectrum.sources[k] == EigenvalueBackwardErrorSource::SingularValue).collect();
        assert!(!singular.is_empty(), "no slot was certified by sigma_min: {:?}", spectrum.sources);
        for k in 0..n {
            assert!(
                spectrum.backward_errors[k] <= spectrum.bands[k],
                "slot {k}: {:e} against band {:e} ({:?})",
                spectrum.backward_errors[k],
                spectrum.bands[k],
                spectrum.sources[k]
            );
        }

        let (_, g_singular, _) = g.svd(false, false).expect("G's singular values");
        let condition = g_singular.iter().fold(0.0_f64, |m, s| m.max(*s)) / g_singular.iter().fold(f64::INFINITY, |m, s| m.min(*s));
        let scale = frobenius(&t);
        let mut planted = vec![(0.5, 0.75), (0.5, 0.75), (2.0, 0.0), (-1.0, 0.0), (-0.625, 0.25)];
        let mut i = 0;
        while i < n {
            let found = (spectrum.re[i], spectrum.im[i]);
            let bar = condition * spectrum.bands[i] * (scale + found.0.hypot(found.1));
            let position = planted
                .iter()
                .position(|&(pr, pi)| (pr - found.0).hypot(pi - found.1) <= bar)
                .unwrap_or_else(|| panic!("eigenvalue {found:?} is within {bar:.3e} of no planted one: {planted:?}"));
            planted.remove(position);
            i += if spectrum.im[i] == 0.0 { 1 } else { 2 };
        }
        assert!(planted.is_empty(), "planted eigenvalues left unrecovered: {planted:?}");
    }

    /// Jordan blocks certify. A 4×4 Jordan block, and the same block with 1e-12 added
    /// at (3, 0), have computed eigenvalues that are exact eigenvalues of a matrix
    /// within the band, so no backward-error certificate refuses them, whatever their
    /// forward error. The perturbed block's eigenvalues split by about 1e-3
    /// (probe job 1209094). A consumer that needs forward accuracy needs a condition
    /// estimate, not this certificate.
    #[test]
    fn real_general_spectrum_certifies_backward_exact_jordan_blocks_2627() {
        let mut jordan = Array2::<f64>::zeros((4, 4));
        for i in 0..4 {
            jordan[[i, i]] = 1.0;
            if i + 1 < 4 {
                jordan[[i, i + 1]] = 1.0;
            }
        }
        let mut perturbed = jordan.clone();
        perturbed[[3, 0]] = 1.0e-12;
        for (label, a) in [("jordan", &jordan), ("perturbed jordan", &perturbed)] {
            let spectrum = real_general_spectrum(a).unwrap_or_else(|error| panic!("{label}: {error}"));
            for k in 0..4 {
                assert!(spectrum.backward_errors[k] <= spectrum.bands[k], "{label}: slot {k}");
            }
        }
        let refusal = FaerLinalgError::GeneralEigenCertificateRefused {
            arm: "eigenvalue backward error sigma_min(A - lambda I)",
            slot: 2,
            measured: 4.5e-2,
            band: 4.7e-12,
        };
        let text = refusal.to_string();
        assert!(text.contains("sigma_min") && text.contains("slot 2"), "the refusal names its arm and slot: {text}");
    }
}

#[cfg(test)]
mod lblt_inertia_2901_tests {
    use super::*;

    /// #2901: the inertia counts a 2×2 pivot block by its eigenvalues. `[[0, 1], [1, 0]]`
    /// admits no 1×1 pivot and has eigenvalues ±1, and `diag(3, −2, 0)` has one
    /// eigenvalue of each sign.
    #[test]
    fn the_bunch_kaufman_inertia_counts_each_pivot_block_by_its_eigenvalues_2901() {
        let swap = Mat::<f64>::from_fn(2, 2, |i, j| if i == j { 0.0 } else { 1.0 });
        let inertia = FaerLblt::new(swap.as_ref(), Side::Lower).inertia();
        assert_eq!((inertia.negative, inertia.zero, inertia.positive), (1, 0, 1), "{inertia:?}");
        assert!((inertia.smallest_pivot + 1.0).abs() <= 4.0 * f64::EPSILON, "{inertia:?}");

        let diagonal = [3.0, -2.0, 0.0];
        let mixed = Mat::<f64>::from_fn(3, 3, |i, j| if i == j { diagonal[i] } else { 0.0 });
        let inertia = FaerLblt::new(mixed.as_ref(), Side::Lower).inertia();
        assert_eq!((inertia.negative, inertia.zero, inertia.positive), (1, 1, 1), "{inertia:?}");
        assert_eq!(inertia.smallest_pivot, -2.0, "{inertia:?}");
    }
}
