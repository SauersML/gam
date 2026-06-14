//! Newton-direction solving: dense/sparse XtWx backends, outer-upper
//! accumulation, positive-definite ridge repair, the implicit/lower-bound
//! Newton solvers, soft-acceptance, and the subsystem direction solve.

use super::*;

const DENSE_OUTER_MAX_P: usize = 1024;


// Estimated FLOP threshold below which spawning rayon workers for the dense
// outer-product path costs more than the work itself. Calibrated to cover
// rayon's per-task overhead (microseconds) plus the cost of zeroing one dense
// buffer per worker; below this, everything stays on the calling thread.
const DENSE_OUTER_PARALLEL_FLOP_THRESHOLD: u64 = 100_000;


/// Backend selection for sparse-design XᵀWX assembly.
///
/// XᵀWX = Σᵢ wᵢ · xᵢ xᵢᵀ. The matrix is symmetric, so only the upper triangle
/// needs to be computed; the only consumer (`assemble_upper`) filters to
/// row ≤ col. Two backends trade off in opposite memory regimes:
///
/// * **Dense outer-product** (small p): allocate a dense p×p buffer and
///   accumulate one rank-1 update per data row. Per-row work is nnz(xᵢ)² —
///   for B-spline-style designs this dominates SpGEMM by orders of magnitude.
///
/// * **Sparse SpGEMM** (large p): faer's symbolic + numeric pipeline. Avoids
///   the dense p×p buffer when it would no longer be cache-resident.
enum XtWxBackend {
    Dense(DenseOuterState),
    Sparse(SparseSpGemmState),
}


/// State for the dense outer-product backend.
///
/// `xtwx_dense` is row-major p×p; the inner loop fills only the upper triangle
/// (j ≤ k), exploiting faer's CSC convention that row indices within each
/// column are stored in ascending order. Lower-triangle entries are left at
/// zero — they are written through the scatter to `xtwxvalues` but never read,
/// because `assemble_upper` filters to row ≤ col.
///
/// `thread_buffers` is bounded at exactly `rayon::current_num_threads()` and
/// reused across PIRLS iterations, so allocation cost is amortized across the
/// entire fit rather than paid per call.
struct DenseOuterState {
    xtwx_dense: Array2<f64>,
    thread_buffers: Vec<Array2<f64>>,
}


/// State for the sparse-SpGEMM backend (faer numeric matmul scratch and the
/// pre-scaled (√W)·X factors that feed it).
///
/// `sqrt_weights` caches `√wᵢ` for each finite nonnegative PIRLS working
/// weight row of X. Without it, the right-factor loop would recompute the same
/// sqrt once per nonzero of X (each row weight gets read by every column that
/// has a nonzero in that row), so for an n=400 K · avg-nnz-per-row=10 design
/// that's 4 M sqrts per PIRLS iteration. Precomputing once collapses that to n
/// sqrts and the inner loop becomes a pure multiply.
///
/// This is deliberately separate from REML/Firth's fixed
/// `observation_weight_sqrt` handling in `solver/reml/firth.rs`: this cache
/// materializes the current working-weight Gram factors, while Firth stores
/// case-weight roots so reduced designs can later be mapped back with
/// reciprocal roots.
struct SparseSpGemmState {
    wxvalues: Vec<f64>,
    wx_tvalues: Vec<f64>,
    sqrt_weights: Vec<f64>,
    info: SparseMatMulInfo,
    scratch: MemBuffer,
    par: Par,
}


pub(crate) struct SparseXtWxCache {
    xtwx_symbolic: SymbolicSparseColMat<usize>,
    xtwxvalues: Vec<f64>,
    nrows: usize,
    ncols: usize,
    nnz: usize,
    x_col_ptr: Vec<usize>,
    xrow_idx: Vec<usize>,
    /// CSC of Xᵀ. In CSC, column i of Xᵀ stores the nonzeros of row i of X,
    /// so this doubles as a CSR view of X for row-by-row access in the
    /// dense-outer path.
    x_t_csc: SparseColMat<usize, f64>,
    backend: XtWxBackend,
}


impl SparseXtWxCache {
    fn new(x: &SparseColMat<usize, f64>) -> Result<Self, EstimationError> {
        // For X^T X where X is CSC: X^T is a SparseRowMat, which we need to
        // convert to CSC format for the matmul API.
        let x_t_csc =
            x.as_ref().transpose().to_col_major().map_err(|_| {
                EstimationError::InvalidInput("failed to transpose to CSC".to_string())
            })?;
        let (xtwx_symbolic, info) = sparse_sparse_matmul_symbolic(x_t_csc.symbolic(), x.symbolic())
            .map_err(|_| {
                EstimationError::InvalidInput("failed to build symbolic XtWX cache".to_string())
            })?;
        let xtwxvalues = vec![0.0; xtwx_symbolic.row_idx().len()];

        let backend = if x.ncols() <= DENSE_OUTER_MAX_P {
            XtWxBackend::Dense(DenseOuterState {
                xtwx_dense: Array2::<f64>::zeros((x.ncols(), x.ncols())),
                thread_buffers: Vec::new(),
            })
        } else {
            // SpGEMM scratch is sized for a fixed parallelism handle, so we
            // capture it once at construction; `get_global_parallelism()` is
            // stable for the lifetime of the process.
            let par = get_global_parallelism();
            let scratch = MemBuffer::new(sparse_sparse_matmul_numeric_scratch::<usize, f64>(
                xtwx_symbolic.as_ref(),
                par,
            ));
            XtWxBackend::Sparse(SparseSpGemmState {
                wxvalues: vec![0.0; x.val().len()],
                wx_tvalues: vec![0.0; x_t_csc.val().len()],
                sqrt_weights: vec![0.0; x.nrows()],
                info,
                scratch,
                par,
            })
        };

        Ok(Self {
            xtwx_symbolic,
            xtwxvalues,
            nrows: x.nrows(),
            ncols: x.ncols(),
            nnz: x.val().len(),
            x_col_ptr: x.symbolic().col_ptr().to_vec(),
            xrow_idx: x.symbolic().row_idx().to_vec(),
            x_t_csc,
            backend,
        })
    }

    fn matches(&self, x: &SparseColMat<usize, f64>) -> bool {
        if self.nrows != x.nrows() || self.ncols != x.ncols() || self.nnz != x.val().len() {
            return false;
        }
        let sym = x.symbolic();
        self.x_col_ptr.as_slice() == sym.col_ptr() && self.xrow_idx.as_slice() == sym.row_idx()
    }

    pub(crate) fn compute_numeric(
        &mut self,
        x: &SparseColMat<usize, f64>,
        weights: &Array1<f64>,
    ) -> Result<(), EstimationError> {
        if weights.len() != self.nrows {
            crate::bail_invalid_estim!(
                "weights length {} does not match design rows {}",
                weights.len(),
                self.nrows
            );
        }

        match &mut self.backend {
            XtWxBackend::Dense(state) => {
                state.compute(self.x_t_csc.as_ref(), weights, self.nrows, self.ncols);
                // Scatter the upper triangle of `xtwx_dense` into the
                // symbolic XᵀX pattern. The pattern stores both halves of
                // the symmetric product, but `assemble_upper` (the sole
                // consumer) reads only entries with row ≤ col, so writing
                // the lower half would be wasted work. The unwritten
                // lower-triangle entries of `xtwxvalues` start at zero
                // (from `vec![0.0; …]` at construction) and remain zero
                // throughout this cache's lifetime, since the dense outer
                // product never writes to lower-triangle positions either.
                let col_ptr = self.xtwx_symbolic.col_ptr();
                let row_idx = self.xtwx_symbolic.row_idx();
                let dense = &state.xtwx_dense;
                for col in 0..self.ncols {
                    let start = col_ptr[col];
                    let end = col_ptr[col + 1];
                    for idx in start..end {
                        let row = row_idx[idx];
                        if row <= col {
                            self.xtwxvalues[idx] = dense[[row, col]];
                        }
                    }
                }
            }
            XtWxBackend::Sparse(state) => state.compute(
                x,
                self.x_t_csc.as_ref(),
                weights,
                self.ncols,
                self.xtwx_symbolic.as_ref(),
                &mut self.xtwxvalues,
            ),
        }

        Ok(())
    }
}


impl DenseOuterState {
    /// Compute the upper triangle of XᵀWX = Σᵢ wᵢ · xᵢ xᵢᵀ into
    /// `self.xtwx_dense`.
    ///
    /// Decides serial vs parallel from a cost model on total estimated FLOPs
    /// and the number of available rayon workers. In parallel mode each
    /// worker accumulates into a thread-local p×p buffer (allocated once and
    /// reused across calls); the workers are summed into `xtwx_dense` in
    /// place, preserving its allocation rather than replacing it with a
    /// freshly-allocated reduction result.
    fn compute(
        &mut self,
        x_t: SparseColMatRef<'_, usize, f64>,
        weights: &Array1<f64>,
        n: usize,
        p: usize,
    ) {
        assert_eq!(self.xtwx_dense.dim(), (p, p));
        self.xtwx_dense.fill(0.0);
        if n == 0 || p == 0 {
            return;
        }
        let xtwx_start = std::time::Instant::now();

        // Cost model: per-row outer-product is nnz(xᵢ)². With avg_nnz ≈
        // nnz_total / n, total work ≈ nnz_total² / n. For designs with
        // uniform row support (e.g. B-splines) this proxy is tight; for
        // mixed-support designs it is an order-of-magnitude estimate, which
        // is all we need to gate parallel spawn.
        let nnz_total = x_t.symbolic().row_idx().len() as u64;
        let work = nnz_total
            .saturating_mul(nnz_total)
            .checked_div(n as u64)
            .unwrap_or(u64::MAX);
        let n_threads = rayon::current_num_threads();
        let parallelize = n_threads > 1 && work >= DENSE_OUTER_PARALLEL_FLOP_THRESHOLD;

        if !parallelize {
            accumulate_outer_upper(&mut self.xtwx_dense, x_t, weights, 0..n);
            log::info!(
                "[STAGE] PIRLS dense XᵀWX assembly (serial) n={} p={} flops~{} elapsed={:.3}s",
                n,
                p,
                (n as u64).saturating_mul((p as u64).saturating_mul(p as u64)),
                xtwx_start.elapsed().as_secs_f64(),
            );
            return;
        }

        // Bounded thread allocation: exactly `n_threads` p×p buffers, one
        // per worker, reused across calls.
        if self.thread_buffers.len() != n_threads {
            self.thread_buffers
                .resize_with(n_threads, || Array2::<f64>::zeros((p, p)));
        }
        let chunk = n.div_ceil(n_threads);
        self.thread_buffers
            .par_iter_mut()
            .enumerate()
            .for_each(|(t, buf)| {
                buf.fill(0.0);
                let start = t * chunk;
                let end = (start + chunk).min(n);
                if start < end {
                    accumulate_outer_upper(buf, x_t, weights, start..end);
                }
            });

        // Reduce per-thread buffers into the cached output. The += preserves
        // `xtwx_dense`'s storage; we never reallocate it.
        for buf in &self.thread_buffers {
            self.xtwx_dense += buf;
        }
        log::info!(
            "[STAGE] PIRLS dense XᵀWX assembly (parallel, threads={}) n={} p={} flops~{} elapsed={:.3}s",
            rayon::current_num_threads(),
            n,
            p,
            (n as u64).saturating_mul((p as u64).saturating_mul(p as u64)),
            xtwx_start.elapsed().as_secs_f64(),
        );
    }
}


impl SparseSpGemmState {
    /// Compute XᵀWX into the symbolic-pattern array `xtwxvalues` via faer's
    /// sparse-sparse matmul: XᵀWX = (√W·X)ᵀ · (√W·X).
    fn compute(
        &mut self,
        x: &SparseColMat<usize, f64>,
        x_t: SparseColMatRef<'_, usize, f64>,
        weights: &Array1<f64>,
        p: usize,
        xtwx_symbolic: SymbolicSparseColMatRef<'_, usize>,
        xtwxvalues: &mut [f64],
    ) {
        let n = x_t.ncols();
        assert_eq!(weights.len(), n);
        assert_eq!(self.sqrt_weights.len(), n);

        assert!(
            weights.iter().all(|&w| w.is_finite() && w >= 0.0),
            "SparseSpGemmState::compute requires finite nonnegative PIRLS weights"
        );
        // Cache √w once per row so the inner loops can multiply
        // without repeated sqrt calls. Single owning slice avoids ndarray
        // bounds checks in the hot loops below.
        let sqrt_w = self.sqrt_weights.as_mut_slice();
        for (dst, &w) in sqrt_w.iter_mut().zip(weights.iter()) {
            *dst = w.sqrt();
        }
        let sqrt_w: &[f64] = sqrt_w;

        let x_ref = x.as_ref();
        // Right factor: √W · X, stored in X's CSC sparsity pattern.
        for col in 0..p {
            let rows = x_ref.row_idx_of_col_raw(col);
            let xvals = x_ref.val_of_col(col);
            let range = x_ref.col_range(col);
            let dst = &mut self.wxvalues[range];
            for ((d, &s), row) in dst.iter_mut().zip(xvals.iter()).zip(rows.iter()) {
                *d = s * sqrt_w[row.unbound()];
            }
        }
        // Left factor: (√W · X)ᵀ in X^T's CSC sparsity pattern. X^T's columns
        // correspond to rows of X, so each column scales by √w_row — read
        // straight from the cached slice with no per-column sqrt.
        for col in 0..n {
            let w = sqrt_w[col];
            let xvals = x_t.val_of_col(col);
            let range = x_t.col_range(col);
            let dst = &mut self.wx_tvalues[range];
            for (d, &s) in dst.iter_mut().zip(xvals.iter()) {
                *d = s * w;
            }
        }

        let wx_ref = SparseColMatRef::new(x.symbolic(), &self.wxvalues[..]);
        let wx_t_ref = SparseColMatRef::new(x_t.symbolic(), &self.wx_tvalues[..]);
        let stack = MemStack::new(&mut self.scratch);
        let xtwxmut = SparseColMatMut::new(xtwx_symbolic, xtwxvalues);
        sparse_sparse_matmul_numeric(
            xtwxmut,
            Accum::Replace,
            wx_t_ref,
            wx_ref,
            1.0,
            &self.info,
            self.par,
            stack,
        );
    }
}


/// Accumulate the upper triangle of Σᵢ wᵢ · xᵢ xᵢᵀ over `rows` into `acc`.
///
/// `x_t` is Xᵀ in CSC: column i lists the nonzero columns of row i of X.
/// Faer's CSC convention stores these in ascending order, so iterating
/// `jj < kk` over per-row index pairs gives `j ≤ k` and only ever writes
/// to `acc[[j, k]]` with `j ≤ k` (the upper triangle, including the
/// diagonal at `jj == kk`).
///
/// Inner-loop layout: `acc` is row-major p×p, so row j lives in the
/// contiguous slice `acc_data[j·p .. (j+1)·p]`. We reborrow that slice once
/// per outer-product step — cheaper than ndarray's `row_mut(j).as_slice_mut()`
/// because it skips the per-call stride-validation and contiguity check.
#[inline]
fn accumulate_outer_upper(
    acc: &mut Array2<f64>,
    x_t: SparseColMatRef<'_, usize, f64>,
    weights: &Array1<f64>,
    rows: std::ops::Range<usize>,
) {
    assert_eq!(acc.nrows(), acc.ncols());
    let p = acc.ncols();
    let acc_data = acc
        .as_slice_mut()
        .expect("dense XᵀWX accumulator is row-major and contiguous");

    for i in rows {
        // Sparse PIRLS precompute deliberately clips to Fisher-style
        // nonnegative weights before the row outer product. The shared REML
        // dense helper preserves signed observed-Hessian weights exactly, so
        // routing this sparse path through it would change curvature semantics.
        let w_i = weights[i].max(0.0);
        if w_i == 0.0 {
            continue;
        }
        let cols = x_t.row_idx_of_col_raw(i);
        let vals = x_t.val_of_col(i);
        let nnz_i = cols.len();
        for jj in 0..nnz_i {
            let j = cols[jj].unbound();
            let wvj = w_i * vals[jj];
            let row = &mut acc_data[j * p..j * p + p];
            for kk in jj..nnz_i {
                let k = cols[kk].unbound();
                row[k] += wvj * vals[kk];
            }
        }
    }
}


pub(crate) fn compute_jeffreys_pirls_diagnostics_sparse(
    link: &InverseLink,
    x_design_csr: &SparseRowMat<usize, f64>,
    eta: ArrayView1<f64>,
    observation_weights: ArrayView1<f64>,
) -> Result<(Array1<f64>, f64, Array1<f64>), EstimationError> {
    let n = x_design_csr.nrows();
    let p = x_design_csr.ncols();
    let mut x_dense = Array2::<f64>::zeros((n, p));
    let xview = x_design_csr.as_ref();
    for i in 0..n {
        let vals = xview.val_of_row(i);
        let cols = xview.col_idx_of_row_raw(i);
        if cols.len() != vals.len() {
            crate::bail_invalid_estim!(
                "sparse row structure mismatch: column/value lengths differ"
            );
        }
        for (idx, &col) in cols.iter().enumerate() {
            x_dense[[i, col.unbound()]] = vals[idx];
        }
    }
    compute_jeffreys_pirls_diagnostics(link, x_dense.view(), eta, observation_weights)
}


pub(crate) fn compute_jeffreys_pirls_diagnostics(
    link: &InverseLink,
    x_design: ArrayView2<f64>,
    eta: ArrayView1<f64>,
    observation_weights: ArrayView1<f64>,
) -> Result<(Array1<f64>, f64, Array1<f64>), EstimationError> {
    // PIRLS must use the same identifiable-subspace Jeffreys functional as the
    // outer REML code:
    //   Φ(β) = 0.5 log|Xᵀ W(η) X|_+.
    // The operator below is the single source of truth for the Jeffreys scalar
    // value, the PIRLS hat-diagonal, AND the working-response score shift the
    // inner solve applies. The Fisher working weight `W(η)` is evaluated for the
    // resolved inverse link; `StandardLink::Logit` reproduces the released logit
    // diagnostics exactly while non-canonical links (probit, cloglog) get the
    // correct link-general shift instead of the logit-pinned `(½ − μ)` term.
    let op = FirthDenseOperator::build_with_observation_weights_for_link(
        link,
        &x_design.to_owned(),
        &eta.to_owned(),
        observation_weights,
    )?;
    Ok((
        op.pirls_hat_diag(),
        op.jeffreys_logdet(),
        op.pirls_firth_score_shift(),
    ))
}


pub(crate) fn ensure_positive_definitewithridge(
    hess: &mut Array2<f64>,
    label: &str,
) -> Result<f64, EstimationError> {
    let ridge = if FIXED_STABILIZATION_RIDGE > 0.0 {
        FIXED_STABILIZATION_RIDGE
    } else {
        0.0
    };

    if hess.cholesky(Side::Lower).is_ok() {
        return Ok(0.0);
    }

    if ridge > 0.0 {
        for i in 0..hess.nrows() {
            hess[[i, i]] += ridge;
        }

        if hess.cholesky(Side::Lower).is_ok() {
            log::debug!("{} stabilized with fixed ridge {:.1e}.", label, ridge);
            return Ok(ridge);
        }
    }

    if let Ok((evals, _)) = hess.eigh(Side::Lower) {
        let min_eig = evals.iter().fold(f64::INFINITY, |a, &b| a.min(b));
        return Err(EstimationError::HessianNotPositiveDefinite {
            min_eigenvalue: min_eig,
        });
    }
    Err(EstimationError::HessianNotPositiveDefinite {
        min_eigenvalue: f64::NEG_INFINITY,
    })
}


pub(crate) fn solve_newton_direction_dense(
    hessian: &Array2<f64>,
    gradient: &Array1<f64>,
    direction_out: &mut Array1<f64>,
) -> Result<(), EstimationError> {
    solve_newton_direction_dense_with_factor(hessian, gradient, direction_out).map(|_| ())
}


pub(super) fn solve_direction_with_dense_factor(
    factor: &FaerSymmetricFactor,
    gradient: &Array1<f64>,
    direction_out: &mut Array1<f64>,
) {
    if direction_out.len() != gradient.len() {
        *direction_out = Array1::zeros(gradient.len());
    }
    direction_out.assign(gradient);
    let mut rhsview = array1_to_col_matmut(direction_out);
    factor.solve_in_place(rhsview.as_mut());
    direction_out.mapv_inplace(|v| -v);
}


/// Fixes the audit-revised geodesic-acceleration note: expose the dense
/// factor so the optional second-order correction can reuse it instead of
/// refactorizing the same Hessian.
pub(super) fn solve_newton_direction_dense_with_factor(
    hessian: &Array2<f64>,
    gradient: &Array1<f64>,
    direction_out: &mut Array1<f64>,
) -> Result<Option<FaerSymmetricFactor>, EstimationError> {
    let dense_solve_start = std::time::Instant::now();
    let p = hessian.nrows();
    if direction_out.len() != gradient.len() {
        *direction_out = Array1::zeros(gradient.len());
    }

    if crate::gpu::cuda_selected() {
        let rhs = Array2::from_shape_vec((p, 1), gradient.to_vec()).map_err(|e| {
            EstimationError::InvalidInput(format!("CUDA PIRLS RHS layout failed: {e}"))
        })?;
        let (solved, _) =
            crate::solver::gpu::pirls_gpu::cholesky_solve_gpu(hessian.view(), rhs.view())
                .map_err(EstimationError::InvalidInput)?;
        direction_out.assign(&solved.column(0));
        direction_out.mapv_inplace(|v| -v);
        if array_is_finite(direction_out) {
            log::info!(
                "[STAGE] PIRLS dense newton solve backend=CUDA p={} flops~{} elapsed={:.3}s route=\"cuSOLVER potrf/potrs\"",
                p,
                (p as u64).saturating_mul((p as u64).saturating_mul(p as u64)) / 3,
                dense_solve_start.elapsed().as_secs_f64(),
            );
            return Ok(None);
        }
    }

    let cpu_route = String::from("CPU stable solver");

    let factor = StableSolver::new("pirls newton direction")
        .factorize(hessian)
        .map_err(EstimationError::LinearSystemSolveFailed)?;
    solve_direction_with_dense_factor(&factor, gradient, direction_out);

    // Validate: bare Cholesky on a near-singular H produces huge spurious
    // step magnitudes in the null direction. If `‖H·δ + g‖∞ / (1+‖g‖∞)` is
    // not small the H is rank-deficient (eigenvalue below floating-point
    // resolution); fall through to the rank-revealing pseudoinverse path
    // which projects rhs onto range(H) before inverting and zeroes the
    // null-direction component of δ. This is the same arithmetic the
    // outer IFT correction uses via penalty_subspace_trace.
    let validation_residual = {
        let h_delta = hessian.dot(direction_out);
        h_delta
            .iter()
            .zip(gradient.iter())
            .map(|(h, g)| (h + g).abs())
            .fold(0.0_f64, f64::max)
    };
    let g_inf = gradient.iter().map(|v| v.abs()).fold(0.0_f64, f64::max);
    let rel = validation_residual / (1.0 + g_inf);
    if !rel.is_finite() || rel > 1.0e-3 {
        // Construct rhs = -gradient (note the gradient is the un-negated
        // ∇f at β; the Newton equation is H·δ = -g) and reach for the
        // pseudoinverse path. `solve_with_pseudoinverse_fallback` handles
        // its own ridge retries and falls back to truncated-eigh
        // pseudoinverse if Cholesky residual is high.
        let rhs = gradient.mapv(|v| -v);
        if let Some(pseudo) = StableSolver::new("pirls newton direction (pseudoinverse fallback)")
            .solve_with_pseudoinverse_fallback(hessian, &rhs, 1.0e-10, 1.0e-3, 1.0e-10)
        {
            direction_out.assign(&pseudo);
            log::info!(
                "[STAGE] PIRLS dense newton solve backend=CPU p={} elapsed={:.3}s route=\"{} + pseudoinverse fallback (rel={:.3e} > 1e-3)\"",
                p,
                dense_solve_start.elapsed().as_secs_f64(),
                cpu_route,
                rel,
            );
            return Ok(Some(factor));
        }
    }
    if array_is_finite(direction_out) {
        log::info!(
            "[STAGE] PIRLS dense newton solve backend=CPU p={} flops~{} elapsed={:.3}s route=\"{}\"",
            p,
            (p as u64).saturating_mul((p as u64).saturating_mul(p as u64)) / 3,
            dense_solve_start.elapsed().as_secs_f64(),
            cpu_route,
        );
        return Ok(Some(factor));
    }
    Err(EstimationError::LinearSystemSolveFailed(
        FaerLinalgError::FactorizationFailed {
            context: "PIRLS dense newton solve exhausted",
        },
    ))
}


/// Solve the Newton direction implicitly via PCG against an operator-form
/// Hessian. Bypasses materialization of the `p × p` Hessian when at least one
/// penalty is operator-form and `p` is large enough that the implicit-matvec
/// cost amortizes against avoiding a dense Cholesky.
///
/// `apply_xtwx`: closure computing `(X^T W X) v`.
/// `xtwx_diag`: diagonal of `X^T W X`, used in the Jacobi preconditioner.
/// `dense_penalties`: pairs `(λ_k, S_k)` for penalties whose dense matrix is
/// the only available representation; their contribution to `H v` is computed
/// as `λ_k · S_k.dot(v)` and their diagonal contribution to the preconditioner
/// is `λ_k · diag(S_k)`.
/// `op_penalties`: pairs `(λ_k, op)` for penalties carrying a `PenaltyOp`
/// handle; their contribution to `H v` is `λ_k · op.matvec(v)` and their
/// diagonal is `λ_k · op.diag()`.
/// `ridge`: nonnegative ridge added to the Hessian diagonal for stabilization.
///
/// On success the negated solution `−H⁻¹ g` is written into `direction_out`,
/// matching the sign convention of `solve_newton_direction_dense`.
pub fn solve_newton_direction_implicit<F>(
    apply_xtwx: F,
    xtwx_diag: ArrayView1<'_, f64>,
    dense_penalties: &[(f64, &Array2<f64>)],
    op_penalties: &[(f64, &dyn crate::terms::penalty_op::PenaltyOp)],
    gradient: &Array1<f64>,
    direction_out: &mut Array1<f64>,
    ridge: f64,
    rel_tol: f64,
    max_iter: usize,
) -> Result<(), EstimationError>
where
    F: Fn(&Array1<f64>) -> Array1<f64>,
{
    let p = gradient.len();
    if xtwx_diag.len() != p {
        crate::bail_invalid_estim!(
            "solve_newton_direction_implicit: xtwx_diag length {} != gradient length {}",
            xtwx_diag.len(),
            p
        );
    }
    for (_, s) in dense_penalties.iter() {
        if s.nrows() != p || s.ncols() != p {
            crate::bail_invalid_estim!(
                "solve_newton_direction_implicit: dense penalty dim {}×{} != p={}",
                s.nrows(),
                s.ncols(),
                p
            );
        }
    }
    for (_, op) in op_penalties.iter() {
        if op.dim() != p {
            crate::bail_invalid_estim!(
                "solve_newton_direction_implicit: op penalty dim {} != p={}",
                op.dim(),
                p
            );
        }
    }
    if direction_out.len() != p {
        *direction_out = Array1::zeros(p);
    }

    let pcg_start = std::time::Instant::now();

    let mut precond_diag = xtwx_diag.to_owned();
    if ridge > 0.0 {
        precond_diag.mapv_inplace(|d| d + ridge);
    }
    for (lambda, s) in dense_penalties.iter() {
        if *lambda == 0.0 {
            continue;
        }
        for i in 0..p {
            precond_diag[i] += *lambda * s[[i, i]];
        }
    }
    for (lambda, op) in op_penalties.iter() {
        if *lambda == 0.0 {
            continue;
        }
        let d = op.diag();
        for i in 0..p {
            precond_diag[i] += *lambda * d[i];
        }
    }

    // SAFETY: `apply_xtwx`, `dense_penalties`, and `op_penalties` are passed
    // by reference into the closure. The PCG closure runs synchronously within
    // this function, so the borrows live for the duration of the call.
    let apply_h = |v: &Array1<f64>| -> Array1<f64> {
        let mut hv = apply_xtwx(v);
        if ridge > 0.0 {
            hv.zip_mut_with(v, |h, &x| *h += ridge * x);
        }
        for (lambda, s) in dense_penalties.iter() {
            if *lambda == 0.0 {
                continue;
            }
            let sv = fast_av(s, v);
            hv.scaled_add(*lambda, &sv);
        }
        for (lambda, op) in op_penalties.iter() {
            if *lambda == 0.0 {
                continue;
            }
            let mut sv = Array1::<f64>::zeros(p);
            op.matvec(v.view(), sv.view_mut());
            hv.scaled_add(*lambda, &sv);
        }
        hv
    };

    let solution =
        crate::linalg::utils::solve_spd_pcg(apply_h, gradient, &precond_diag, rel_tol, max_iter)
            .ok_or(EstimationError::LinearSystemSolveFailed(
                FaerLinalgError::FactorizationFailed {
                    context: "PIRLS implicit PCG solve exhausted",
                },
            ))?;

    direction_out.assign(&solution);
    direction_out.mapv_inplace(|v| -v);
    if !array_is_finite(direction_out) {
        return Err(EstimationError::LinearSystemSolveFailed(
            FaerLinalgError::FactorizationFailed {
                context: "PIRLS implicit PCG non-finite direction",
            },
        ));
    }
    log::info!(
        "[STAGE] PIRLS implicit (PCG) newton solve p={} dense_pens={} op_pens={} elapsed={:.3}s",
        p,
        dense_penalties.len(),
        op_penalties.len(),
        pcg_start.elapsed().as_secs_f64(),
    );
    Ok(())
}


pub(super) fn project_coefficients_to_lower_bounds(
    beta: &mut Array1<f64>,
    lower_bounds: &Array1<f64>,
) {
    for i in 0..beta.len() {
        let lb = lower_bounds[i];
        if lb.is_finite() && beta[i] < lb {
            beta[i] = lb;
        }
    }
}


/// Compute the projected gradient norm for bound-constrained optimization.
///
/// At a constrained optimum, gradient components for variables at their lower
/// bound that point into the infeasible direction (gradient > 0 for minimization)
/// are KKT multipliers, not convergence defects.  Zeroing them gives the
/// standard "projected gradient" used to test stationarity.
/// Relative and absolute tolerances for deciding when a coefficient sits "at"
/// its lower bound (an active box constraint). A coefficient is active when its
/// slack is below `ACTIVE_BOUND_REL_TOL * scale + ACTIVE_BOUND_ABS_TOL`; the
/// absolute term keeps genuinely-near-zero bounded coefficients (e.g. I-spline
/// time coefficients pinned around 1e-6) from being treated as interior. Both
/// the projected-gradient norm and the active-set classifier must use the same
/// band so KKT diagnostics and the working set agree.
const ACTIVE_BOUND_REL_TOL: f64 = 1e-6;

const ACTIVE_BOUND_ABS_TOL: f64 = 1e-10;


pub(crate) fn projected_gradient_norm(
    gradient: &Array1<f64>,
    beta: &Array1<f64>,
    lower_bounds: Option<&Array1<f64>>,
) -> f64 {
    let Some(lb) = lower_bounds else {
        return gradient.dot(gradient).sqrt();
    };
    let mut sum_sq = 0.0;
    for i in 0..gradient.len() {
        let g = gradient[i];
        if lb[i].is_finite() && g > 0.0 {
            // Use a relative+absolute tolerance so near-bound coefficients
            // (e.g. I-spline time coefficients at 1e-6) are recognized as
            // active.  At a KKT point the gradient into the infeasible region
            // is a multiplier, not a convergence defect.
            let slack = beta[i] - lb[i];
            let scale = beta[i].abs().max(lb[i].abs()).max(1.0);
            let tol = ACTIVE_BOUND_REL_TOL * scale + ACTIVE_BOUND_ABS_TOL;
            if slack < tol {
                continue;
            }
        }
        sum_sq += g * g;
    }
    sum_sq.sqrt()
}


/// "Soft" P-IRLS acceptance reasons — fits that did not certify strict KKT
/// stationarity but that the post-loop rescue would still classify as
/// `StalledAtValidMinimum`. Evaluating them per-iter (gated by a streak)
/// lets the loop exit at the iteration that first meets the criterion
/// instead of grinding to `MaxIterations` only to be rescued with the
/// same conditions.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) enum PirlsSoftAccept {
    /// Projected gradient inside the 10× near-stationary band AND the
    /// progress signal has plateaued at `tol · objective_scale` (or, in
    /// the LM-rejection context, at the much tighter `1e-12 · |Φ|` model
    /// noise floor — see [`SoftAcceptProgress`]). The standard
    /// "good-enough plateau" rescue, and the only branch that fires
    /// when no LM step was accepted.
    NearStationaryPlateau,
    /// `max|η|` is pinned against [`PIRLS_ETA_ABS_CAP`] AND the deviance
    /// has plateaued. Same saturated-boundary class as separated binomial
    /// fits: extra Newton work only re-tries the clipped boundary. Only
    /// meaningful when a step was actually taken — the LM-rejection
    /// context skips this branch.
    BoundarySaturation,
    /// Projected gradient is small *relative to the objective magnitude*
    /// (not just the dimension scale) AND the deviance has plateaued
    /// strictly (×0.1 floor) AND is non-decreasing. This is the
    /// per-observation rescue for large-scale GLMs where ‖g‖ scales
    /// with √n and the absolute KKT test becomes systematically too
    /// tight even when the fit is functionally converged. Like
    /// [`PirlsSoftAccept::BoundarySaturation`], this is only meaningful
    /// when a step was actually taken.
    RelativeBandPlateau,
}


/// Source of the "is the fit still moving?" signal handed to
/// [`pirls_soft_acceptance`]. There are two contexts in which we need to
/// decide whether a fit should be accepted as a soft minimum:
///
/// - [`SoftAcceptProgress::Realized`] — a step was accepted (per-iter
///   path) or the loop has run out of iterations (post-loop rescue). We
///   know the realized change in penalized deviance and can compare it
///   directly against the standard `tol · objective_scale` plateau band.
///   All three [`PirlsSoftAccept`] branches are eligible.
///
/// - [`SoftAcceptProgress::Predicted`] — no LM candidate step survived
///   screening, so there is no realized Δdev to test. Instead, the
///   model's *predicted* reduction from the unaccepted step (`predicted
///   = -(g·d + ½ d·H·d)`) is compared against the much tighter model
///   noise floor `1e-12 · max(|Φ|, 1)`. This preserves the historical
///   LM-rejection acceptance criterion exactly: only the
///   near-stationary-plateau branch is eligible (saturated-η and
///   relative-band tests both rely on a realized deviance change and
///   would widen acceptance if applied with `predicted=0`).
#[derive(Clone, Copy, Debug)]
pub(super) enum SoftAcceptProgress {
    /// Realized change in penalized deviance from the most recent
    /// accepted step (per-iter) or final accepted step (post-loop).
    Realized { dev_change: f64 },
    /// Predicted reduction `-(g·d + ½ d·H·d)` from the unaccepted LM
    /// candidate step, paired with the current penalized objective so
    /// the helper can scale the model noise floor consistently with the
    /// LM-rejection branch's historical `1e-12 · max(|Φ|, 1)` cutoff.
    Predicted {
        predicted_reduction: f64,
        current_penalized: f64,
    },
}


/// Evaluate every "soft" acceptance criterion that the post-loop rescue
/// applies to a fit which has hit `MaxIterations`. Returns the first
/// matching reason, or `None` if no criterion fires.
///
/// Three call sites share this helper:
///
/// 1. **Per-iter** (after an accepted step) — gated on a 2-iter plateau
///    streak so a single noisy step that briefly satisfies the band
///    can't trigger an early exit. All three branches are eligible.
/// 2. **Post-loop rescue** (MaxIterations hit) — accepts immediately;
///    all three branches are eligible.
/// 3. **LM-rejection** (no candidate step survived screening) — accepts
///    immediately, but only the [`PirlsSoftAccept::NearStationaryPlateau`]
///    branch is eligible, with the tighter model noise floor that the
///    historical LM-rejection check used. Saturated-η and relative-band
///    tests need a realized Δdev and are skipped.
///
/// Sharing the helper guarantees the three acceptance contexts stay in
/// lockstep — anything accepted post-loop is also a candidate for
/// early-exit, and the LM-rejection branch accepts exactly the same set
/// of states it accepted before unification.
#[inline]
pub(super) fn pirls_soft_acceptance(
    state: &WorkingState,
    projected_grad: f64,
    progress: SoftAcceptProgress,
    max_abs_eta: f64,
    progress_tol: f64,
    kkt_tol: f64,
) -> Option<PirlsSoftAccept> {
    let objective_scale = state.deviance.abs().max(state.penalty_term.abs()).max(1.0);
    // Progress tests stay on the fixed PIRLS tolerance; only KKT stationarity uses kkt_tol.
    let scaled_dev_tol = progress_tol * objective_scale;

    // Near-stationary plateau is eligible in every context. The only
    // thing that varies is which "is the fit still moving?" signal we
    // compare against which floor.
    let near_stationary_plateau = match progress {
        SoftAcceptProgress::Realized { dev_change } => {
            state.near_stationary_kkt(projected_grad, kkt_tol) && dev_change.abs() < scaled_dev_tol
        }
        SoftAcceptProgress::Predicted {
            predicted_reduction,
            current_penalized,
        } => {
            // Historical LM-rejection floor: model-predicted reduction
            // below `1e-12 · max(|Φ|, 1)` is indistinguishable from
            // numerical noise on the quadratic model. Keep this exact
            // formula — it is strictly tighter than `tol · scaled_dev_tol`
            // for the standard tol=1e-6, so the unified helper does not
            // widen the LM-rejection acceptance set.
            let reduction_noise_floor = current_penalized.abs().max(1.0) * 1e-12;
            state.near_stationary_kkt(projected_grad, kkt_tol)
                && predicted_reduction.abs() <= reduction_noise_floor
        }
    };
    if near_stationary_plateau {
        return Some(PirlsSoftAccept::NearStationaryPlateau);
    }

    // The remaining branches both require a realized Δdev to be
    // meaningful: η-cap saturation tests "did the step move and yet η
    // stayed pinned at the cap?", and the relative-band plateau tests a
    // signed, magnitude-bounded Δdev. Substituting `predicted=0` would
    // trivially satisfy both with zero diagnostic value and would widen
    // the LM-rejection acceptance set, so they are gated on a Realized
    // progress signal.
    let dev_change = match progress {
        SoftAcceptProgress::Realized { dev_change } => dev_change,
        SoftAcceptProgress::Predicted { .. } => return None,
    };

    if max_abs_eta >= PIRLS_ETA_ABS_CAP * (1.0 - 1e-12) && dev_change.abs() < scaled_dev_tol {
        return Some(PirlsSoftAccept::BoundarySaturation);
    }

    if projected_grad <= progress_tol.max(1e-6) * objective_scale
        && dev_change.abs() < scaled_dev_tol * 0.1
        && dev_change >= 0.0
    {
        return Some(PirlsSoftAccept::RelativeBandPlateau);
    }

    None
}


pub(super) fn constrained_stationarity_norm(
    gradient: &Array1<f64>,
    beta: &Array1<f64>,
    lower_bounds: Option<&Array1<f64>>,
    linear_constraints: Option<&LinearInequalityConstraints>,
) -> f64 {
    // `gradient`, `beta`, and `linear_constraints` are all represented in the
    // current PIRLS coefficient basis (raw sparse-native or Qs-transformed).
    // At an active inequality, the raw gradient can carry a valid KKT
    // multiplier, so convergence must use the full KKT residual in that same
    // frame rather than the unprojected gradient norm.
    if let Some(constraints) = linear_constraints {
        let kkt = compute_constraint_kkt_diagnostics(beta, gradient, constraints);
        return kkt
            .primal_feasibility
            .max(kkt.dual_feasibility)
            .max(kkt.complementarity)
            .max(kkt.stationarity);
    }
    projected_gradient_norm(gradient, beta, lower_bounds)
}


fn solve_subsystem_direction(
    h_sub: ndarray::ArrayView2<f64>,
    g_sub: ndarray::ArrayView1<f64>,
    out: &mut Array1<f64>,
) -> Result<(), EstimationError> {
    let n = g_sub.len();
    if out.len() != n {
        *out = Array1::zeros(n);
    }
    // Try direct factorization first.
    if let Ok(factor) = StableSolver::new("pirls bounded subsystem").factorize_any(&h_sub) {
        out.assign(&g_sub);
        let mut rhs = array1_to_col_matmut(out);
        factor.solve_in_place(rhs.as_mut());
        out.mapv_inplace(|v| -v);
        if array_is_finite(out) {
            return Ok(());
        }
    }
    // Factorization failed or produced non-finite values — the reduced Hessian
    // is singular or nearly so (common on underdetermined problems).  Add a
    // diagonal ridge and retry with geometrically increasing strength.
    let diag_scale = (0..n)
        .map(|i| h_sub[[i, i]].abs())
        .fold(0.0_f64, f64::max)
        .max(1.0);
    let mut tau = 1e-8 * diag_scale;
    let mut h_reg = h_sub.to_owned();
    for _ in 0..12 {
        for i in 0..n {
            h_reg[[i, i]] = h_sub[[i, i]] + tau;
        }
        if let Ok(factor) = StableSolver::new("pirls bounded subsystem ridge").factorize(&h_reg) {
            out.assign(&g_sub);
            let mut rhs = array1_to_col_matmut(out);
            factor.solve_in_place(rhs.as_mut());
            out.mapv_inplace(|v| -v);
            if array_is_finite(out) {
                return Ok(());
            }
        }
        tau *= 10.0;
    }
    // All ridge attempts failed — fall back to steepest descent on the
    // free subspace: d = -g / ||g||, scaled to a conservative step.
    let gnorm = g_sub.dot(&g_sub).sqrt();
    if gnorm > 0.0 {
        let scale = 1.0 / gnorm.max(diag_scale);
        for i in 0..n {
            out[i] = -g_sub[i] * scale;
        }
        return Ok(());
    }
    // Zero gradient — already at optimum on this subspace.
    out.fill(0.0);
    Ok(())
}


pub(super) fn linear_constraints_from_lower_bounds(
    lower_bounds: &Array1<f64>,
) -> Option<LinearInequalityConstraints> {
    LinearInequalityConstraints::from_per_coordinate_lower_bounds(lower_bounds)
}


pub(crate) fn compute_constraint_kkt_diagnostics(
    beta: &Array1<f64>,
    gradient: &Array1<f64>,
    constraints: &LinearInequalityConstraints,
) -> ConstraintKktDiagnostics {
    active_set::compute_constraint_kkt_diagnostics(beta, gradient, constraints)
}


/// Select which active bound-constraint to release in the primal active-set
/// QP loop, or `None` when KKT is satisfied (no negative multiplier).
///
/// `use_blands` switches between two pivoting rules with the same KKT-test
/// semantics but different anti-cycling guarantees:
///
/// - `false` — **worst-violation**: release the constraint with the most
///   negative multiplier `λ_i = g_i + (H d)_i`. Greedy and fast on
///   non-degenerate problems but can cycle when several constraints have
///   multipliers near zero of comparable magnitude.
/// - `true` — **Bland's rule**: release the *lowest-index* constraint with a
///   strictly-negative multiplier (using a scale-aware deadband to ignore
///   pure round-off). This is the textbook anti-cycling choice — combined
///   with Bland-compatible tie-breaking on entering, it guarantees the
///   active-set sequence visits each vertex at most once and so terminates
///   in finitely many pivots.
pub(crate) fn select_active_set_release(
    gradient: &Array1<f64>,
    hd: &Array1<f64>,
    active_idx: &[usize],
    use_blands: bool,
) -> Option<usize> {
    if use_blands {
        for &i in active_idx {
            let lambda_i = gradient[i] + hd[i];
            let scale = gradient[i].abs().max(hd[i].abs()).max(1.0);
            let tol = 64.0 * f64::EPSILON * scale;
            if lambda_i < -tol {
                return Some(i);
            }
        }
        None
    } else {
        let mut worst = 0.0_f64;
        let mut idx = None;
        for &i in active_idx {
            let lambda_i = gradient[i] + hd[i];
            if lambda_i < worst {
                worst = lambda_i;
                idx = Some(i);
            }
        }
        idx
    }
}


pub(crate) fn solve_newton_directionwith_lower_bounds(
    hessian: &Array2<f64>,
    gradient: &Array1<f64>,
    beta: &Array1<f64>,
    lower_bounds: &Array1<f64>,
    direction_out: &mut Array1<f64>,
    active_hint: Option<&mut Vec<usize>>,
) -> Result<(), EstimationError> {
    // Bound-constrained Newton step on the local quadratic model:
    //
    //   min_d  g^T d + 0.5 d^T H d
    //   s.t.   beta + d >= l
    //
    // KKT conditions for active bounds A:
    //   d_A = 0,
    //   H_FF d_F = -g_F,
    //   lambda_A = g_A + (H d)_A >= 0.
    //
    // We solve the free subsystem, enforce primal feasibility by clipping to the
    // first boundary hit, then enforce dual feasibility by releasing active bounds
    // with negative multipliers. This is the standard primal active-set loop for
    // strictly convex box QPs.
    let p = gradient.len();
    if lower_bounds.len() != p || beta.len() != p {
        crate::bail_invalid_estim!(
            "lower-bound size mismatch: beta={}, gradient={}, bounds={}",
            beta.len(),
            gradient.len(),
            lower_bounds.len()
        );
    }
    if direction_out.len() != p {
        *direction_out = Array1::zeros(p);
    }
    direction_out.fill(0.0);

    // Fast path: if unconstrained Newton step is already feasible for all lower
    // bounds, it is the exact constrained minimizer (strict convex quadratic).
    let has_active_hint = active_hint
        .as_ref()
        .map(|hint| !hint.is_empty())
        .unwrap_or(false);
    if !has_active_hint && solve_newton_direction_dense(hessian, gradient, direction_out).is_ok() {
        let mut feasible = true;
        for i in 0..p {
            let lb = lower_bounds[i];
            if lb.is_finite() && beta[i] + direction_out[i] < lb {
                feasible = false;
                break;
            }
        }
        if feasible {
            return Ok(());
        }
    }

    let mut active = vec![false; p];
    if let Some(hint) = active_hint.as_ref() {
        for &idx in hint.iter() {
            if idx < p {
                active[idx] = true;
            }
        }
    }
    for i in 0..p {
        let lb = lower_bounds[i];
        if lb.is_finite() && gradient[i] > 0.0 {
            // Use a relative+absolute tolerance matching projected_gradient_norm
            // so coefficients near the bound (e.g. I-spline at 1e-6) with positive
            // gradient (KKT multiplier) are correctly identified as active.
            let scale = beta[i].abs().max(lb.abs()).max(1.0);
            let tol = ACTIVE_BOUND_REL_TOL * scale + ACTIVE_BOUND_ABS_TOL;
            if beta[i] <= lb + tol {
                active[i] = true;
            }
        }
    }

    // Hybrid pivoting: worst-violation gives faster average convergence on
    // non-degenerate problems but can cycle at degenerate vertices (multiple
    // active constraints with multipliers near zero, ping-ponging activate/
    // release of the same coordinate). After a worst-violation grace period
    // we switch to Bland's lowest-index rule, which monotonically orders the
    // active-set sequence visited and therefore terminates in finitely many
    // additional pivots. Entering already uses Bland-compatible tie-breaking
    // (smallest α_hit, ties broken by ascending free-index iteration order
    // because `boundary_hit_step_fraction` requires `step < current_step_limit`
    // strictly), so the leaving rule is the only place anti-cycling has to
    // be enforced.
    const BLANDS_RULE_GRACE: usize = 2;
    let blands_threshold = BLANDS_RULE_GRACE * (p + 1);
    let max_iters = 8 * (p + 1);
    let mut d_free = Array1::<f64>::zeros(p);
    // Reusable hoisted buffers for the free-block Newton subsystem; sliced down
    // to the current `n_free` each iteration to avoid reallocating the p×p
    // block and length-p prefix on every active-set pivot.
    let mut h_ff_buf = Array2::<f64>::zeros((p, p));
    let mut g_f_buf = Array1::<f64>::zeros(p);
    for it in 0..max_iters {
        let use_blands = it >= blands_threshold;
        let free_idx: Vec<usize> = (0..p).filter(|&i| !active[i]).collect();
        let active_idx: Vec<usize> = (0..p).filter(|&i| active[i]).collect();
        direction_out.fill(0.0);
        for &i in &active_idx {
            let lb = lower_bounds[i];
            if lb.is_finite() {
                direction_out[i] = lb - beta[i];
            }
        }
        if free_idx.is_empty() {
            let hd = fast_av(hessian, direction_out);
            if let Some(idx) = select_active_set_release(gradient, &hd, &active_idx, use_blands) {
                active[idx] = false;
                continue;
            }
            if let Some(hint) = active_hint {
                hint.clear();
                hint.extend((0..p).filter(|&i| active[i]));
            }
            return Ok(());
        }

        let n_free = free_idx.len();
        // Reuse hoisted top-left n_free×n_free block and length-n_free prefix.
        {
            let mut h_ff = h_ff_buf.slice_mut(ndarray::s![..n_free, ..n_free]);
            let mut g_f = g_f_buf.slice_mut(ndarray::s![..n_free]);
            for (ii, &i) in free_idx.iter().enumerate() {
                let mut gi = gradient[i];
                for &j in &active_idx {
                    gi += hessian[[i, j]] * direction_out[j];
                }
                g_f[ii] = gi;
                for (jj, &j) in free_idx.iter().enumerate() {
                    h_ff[[ii, jj]] = hessian[[i, j]];
                }
            }
        }
        solve_subsystem_direction(
            h_ff_buf.slice(ndarray::s![..n_free, ..n_free]),
            g_f_buf.slice(ndarray::s![..n_free]),
            &mut d_free,
        )?;
        for (ii, &i) in free_idx.iter().enumerate() {
            direction_out[i] = d_free[ii];
        }

        // Enforce primal feasibility for bound-constrained coefficients.
        let mut hit_idx: Option<usize> = None;
        let mut best_alpha = 1.0_f64;
        for &i in &free_idx {
            let lb = lower_bounds[i];
            if !lb.is_finite() {
                continue;
            }
            let slack = beta[i] - lb;
            let di = direction_out[i];
            if let Some(alpha_i) = boundary_hit_step_fraction(slack, di, best_alpha) {
                best_alpha = alpha_i;
                hit_idx = Some(i);
            }
        }
        if let Some(i_hit) = hit_idx {
            for i in 0..p {
                direction_out[i] *= best_alpha;
            }
            active[i_hit] = true;
            continue;
        }

        // Dual feasibility on active constraints:
        // λ_i = g_i + (H d)_i must be >= 0 for all active lower bounds.
        let hd = fast_av(hessian, direction_out);
        if let Some(idx) = select_active_set_release(gradient, &hd, &active_idx, use_blands) {
            active[idx] = false;
            continue;
        }

        if let Some(hint) = active_hint {
            hint.clear();
            hint.extend((0..p).filter(|&i| active[i]));
        }
        return Ok(());
    }

    // Active-set loop did not converge — fall back to a projected gradient
    // step.  This is always feasible and gives a descent direction, letting the
    // outer LM loop decide whether to accept or increase damping.
    let gnorm = gradient.dot(gradient).sqrt();
    if gnorm > 0.0 {
        let diag_scale = (0..p)
            .map(|i| hessian[[i, i]].abs())
            .fold(0.0_f64, f64::max)
            .max(1.0);
        let step_scale = 1.0 / diag_scale;
        for i in 0..p {
            let di = -gradient[i] * step_scale;
            let lb = lower_bounds[i];
            if lb.is_finite() && beta[i] + di < lb {
                direction_out[i] = lb - beta[i];
            } else {
                direction_out[i] = di;
            }
        }
    } else {
        direction_out.fill(0.0);
    }
    if let Some(hint) = active_hint {
        hint.clear();
    }
    Ok(())
}


/// Reduce a constraint matrix to full row rank using column-pivoted QR on A^T.
///
/// Given k constraint rows in R^p, computes the numerical row rank r via
/// pivoted QR of A^T (p × k) with a tolerance scaled to `eps · max(k, p) ·
/// |R₀₀|`, and retains only the r pivot rows.  Dropped rows have their
/// group membership merged into the most-aligned kept row so that the
/// active-set QP can still release the underlying original constraints via
/// multiplier signs.
///
/// This is a shared numerical primitive used by both the PIRLS and
/// custom-family active-set solvers.
pub(crate) fn solve_newton_directionwith_linear_constraints(
    hessian: &Array2<f64>,
    gradient: &Array1<f64>,
    beta: &Array1<f64>,
    constraints: &LinearInequalityConstraints,
    direction_out: &mut Array1<f64>,
    active_hint: Option<&mut Vec<usize>>,
) -> Result<(), EstimationError> {
    active_set::solve_newton_direction_with_linear_constraints(
        hessian,
        gradient,
        beta,
        constraints,
        direction_out,
        active_hint,
    )
}


// loop_driver owns: default_beta_guess_external, solve_intercept_for_prevalence,
// assemble_pirls_result, detect_logit_instability, stack_lambdaweighted_penalty_root_canonical,
// build_sparse_native_reparam_result, build_diagonal_penalty_from_kronecker, canonical_prior_shift,
// canonical_prior_mean_aggregate, PirlsProblem, PenaltyConfig, fit_model_for_fixed_rho,
// fit_model_for_fixed_rho_with_adaptive_kkt, PirlsConfig, make_reparam_operator,
// build_transformed_lower_bound_constraints*, build_transformed_linear_constraints*,
// merge_linear_constraints, sparse_from_denseview.
