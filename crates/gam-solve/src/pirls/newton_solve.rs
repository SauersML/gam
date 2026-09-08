//! Penalized-Hessian assembly and the Newton / penalized-least-squares step:
//! dense vs sparse XᵀWX Gram backends, ridge positive-definiteness rescue,
//! dense/sparse/implicit Newton-direction solves, bound- and linear-constraint
//! active-set KKT machinery, and the soft-acceptance progress test.

use super::*;
// `Unbind::unbound()` maps a faer bound sparse column index back to `usize`
// for `x_dense` indexing. Imported directly at the call site rather than via
// the pirls prelude re-export (which the deny-warnings build flagged as an
// unused re-export even though the trait method is used here, #2306/build).
use faer::Unbind;

pub(crate) const DENSE_OUTER_MAX_P: usize = 1024;

// Estimated FLOP threshold below which spawning rayon workers for the dense
// outer-product path costs more than the work itself. Calibrated to cover
// rayon's per-task overhead (microseconds) plus the cost of zeroing one dense
// buffer per worker; below this, everything stays on the calling thread.
pub(crate) const DENSE_OUTER_PARALLEL_FLOP_THRESHOLD: u64 = 100_000;

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
pub(crate) enum XtWxBackend {
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
pub(crate) struct DenseOuterState {
    pub(crate) xtwx_dense: Array2<f64>,
    pub(crate) thread_buffers: Vec<Array2<f64>>,
}

/// State for the sparse-SpGEMM backend (faer numeric matmul scratch and the
/// row-scaled `X^T W` left factor and unscaled `X` right factor that feed it).
/// This asymmetric factorization preserves signed observed-information weights
/// exactly and is cheaper than forming two square-root-scaled factors.
pub(crate) struct SparseSpGemmState {
    pub(crate) wxvalues: Vec<f64>,
    pub(crate) wx_tvalues: Vec<f64>,
    pub(crate) info: SparseMatMulInfo,
    pub(crate) scratch: MemBuffer,
    pub(crate) par: Par,
}

pub(crate) struct SparseXtWxCache {
    pub(crate) xtwx_symbolic: SymbolicSparseColMat<usize>,
    pub(crate) xtwxvalues: Vec<f64>,
    pub(crate) nrows: usize,
    pub(crate) ncols: usize,
    pub(crate) nnz: usize,
    pub(crate) x_col_ptr: Vec<usize>,
    pub(crate) xrow_idx: Vec<usize>,
    /// CSC of Xᵀ. In CSC, column i of Xᵀ stores the nonzeros of row i of X,
    /// so this doubles as a CSR view of X for row-by-row access in the
    /// dense-outer path.
    pub(crate) x_t_csc: SparseColMat<usize, f64>,
    pub(crate) backend: XtWxBackend,
}

impl SparseXtWxCache {
    pub(crate) fn new(x: &SparseColMat<usize, f64>) -> Result<Self, EstimationError> {
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

    pub(crate) fn matches(&self, x: &SparseColMat<usize, f64>) -> bool {
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
    pub(crate) fn compute(
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
    /// Compute XᵀWX via faer's sparse-sparse matmul as `(XᵀW) · X`.
    pub(crate) fn compute(
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
        assert!(weights.iter().all(|w| w.is_finite()));

        let x_ref = x.as_ref();
        // Right factor: X, copied into the reusable numeric buffer.
        for col in 0..p {
            let xvals = x_ref.val_of_col(col);
            let range = x_ref.col_range(col);
            let dst = &mut self.wxvalues[range];
            dst.copy_from_slice(xvals);
        }
        // Left factor: X^T W. X^T's columns correspond to rows of X, so each
        // column scales once by the exact (possibly signed) row weight.
        for col in 0..n {
            let w = weights[col];
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
pub(crate) fn accumulate_outer_upper(
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
        let w_i = weights[i];
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

/// Densify a CSR design row-major into a fresh `Array2`. Shared by the sparse
/// Firth diagnostics and the sparse design-factor builder so both densify the
/// design identically.
fn dense_design_from_csr(
    x_design_csr: &SparseRowMat<usize, f64>,
) -> Result<Array2<f64>, EstimationError> {
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
    Ok(x_dense)
}

/// Build the β-independent Firth design factor from a CSR design (#1575). The
/// densified design and the factor it produces are identical to what the
/// per-iteration sparse diagnostics path constructed; only the β-dependent
/// remainder is then rebuilt per Newton iteration.
pub(super) fn build_firth_design_factor_sparse(
    x_design_csr: &SparseRowMat<usize, f64>,
    observation_weights: ArrayView1<f64>,
) -> Result<FirthDesignFactor, EstimationError> {
    let x_dense = dense_design_from_csr(x_design_csr)?;
    FirthDenseOperator::build_design_factor_with_observation_weights(
        &x_dense,
        Some(observation_weights),
    )
}

/// Build the β-independent Firth design factor from a dense design (#1575).
pub(super) fn build_firth_design_factor_dense(
    x_design: ArrayView2<f64>,
    observation_weights: ArrayView1<f64>,
) -> Result<FirthDesignFactor, EstimationError> {
    FirthDenseOperator::build_design_factor_with_observation_weights(
        &x_design.to_owned(),
        Some(observation_weights),
    )
}

/// Compute the Firth working-response diagnostics and the exact Jeffreys
/// coefficient Hessian from one cached β-independent design factor.
///
/// The inner objective is `data + penalty - Φ`, so its Newton curvature is
/// `H₀ - HΦ`, not the Fisher-scoring surrogate `H₀`.  Building the full
/// β-dependent operator here shares the reduced Fisher inverse, leverage, and
/// Hadamard-Gram contraction between the score shift and `HΦ`; the expensive
/// design Gram/eigenspace remains cached in `factor` (#1575).
pub(super) fn jeffreys_pirls_diagnostics_and_hessian_from_factor(
    factor: &FirthDesignFactor,
    link: &InverseLink,
    eta: ArrayView1<f64>,
) -> Result<(Array1<f64>, f64, Array1<f64>, Array2<f64>), EstimationError> {
    let op = FirthDenseOperator::build_from_design_factor(factor, link, &eta.to_owned())?;
    let hat_diag = &op.w * &op.h_diag;
    let mut score_shift = Array1::<f64>::zeros(op.w.len());
    for i in 0..op.w.len() {
        if op.w[i] > 0.0 {
            score_shift[i] = 0.5 * (op.w1[i] / op.w[i]) * op.h_diag[i];
        }
    }
    let diag_term = gam_linalg::faer_ndarray::fast_xt_diag_x(
        &op.x_dense,
        &(&op.w2 * &op.h_diag),
    );
    let bpb = gam_linalg::faer_ndarray::fast_atb(&op.b_base, &op.p_b_base);
    let mut hphi = 0.5 * (diag_term - bpb);
    gam_linalg::matrix::symmetrize_in_place(&mut hphi);
    if !hphi.iter().all(|value| value.is_finite()) {
        crate::bail_invalid_estim!("Firth/Jeffreys coefficient Hessian is non-finite");
    }
    Ok((hat_diag, op.jeffreys_logdet(), score_shift, hphi))
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

    // A non-finite assembly is a different defect class from indefiniteness
    // (and eigh of a NaN-carrying triangle can report arbitrary "positive"
    // spectra); name it precisely instead of letting it masquerade as a
    // not-positive-definite refusal (#2316 triage).
    if !hess.iter().all(|value| value.is_finite()) {
        crate::bail_invalid_estim!(
            "{label}: assembled Hessian contains non-finite entries; refusing to factor"
        );
    }

    // δ IS APPLIED UNCONDITIONALLY HERE. THE HISTORY BELOW IS WHY.
    //
    // This paragraph used to open "δ IS CHOSEN BY A BRANCH HERE, AND THAT IS A
    // KNOWN DEFECT", forty lines above this same block's own statement that δ
    // is now unconditional and above the code that applies it that way. That
    // sentence outlived the defect it described and sent at least one later
    // reader hunting for a branching selector that no longer exists, so it is
    // written in the past tense now.
    //
    // This selector USED TO return `0.0` when the bare factorization succeeded
    // and FIXED_STABILIZATION_RIDGE when it did not, which makes δ a function
    // of ρ — through a Cholesky-success predicate on a near-singular matrix —
    // while δ is carried as `RidgePolicy::exact_full_objective()` and so enters
    // the outer criterion through `0.5·log|H|`. Measured on the #1575
    // binomial/logit fixture: the outer cost at ρ displacements of 1e-9 and
    // 1e-12 differed from its value at ρ₀ by exactly −9.2103400803 and
    // +18.4206788262 — `0.5·ln(1e8)` and `0.5·ln(1e16)` — at identical
    // deviance, edf and penalty term, the whole difference being `ridge = 1e-8`
    // at one point and `ridge = 0` at its neighbours. A criterion that jumps by
    // 9.21 between neighbouring ρ is not a function of ρ, and neither a line
    // search nor a certificate is well posed on it.
    //
    // Applying δ unconditionally removes the jump and fixes both #1575 gates
    // (REML 503.36, edf 18.38, |g| 1.5e-5, 26 inner solves — better than the
    // 2026-07-04 healthy record on every axis). It was landed as `3213e26d3`,
    // REVERTED in `386ba9e37`, and RELANDED in `fc2b286a2` once the companion
    // forms carried δ. The revert happened because an unconditional δ also
    // makes every companion form that assumes δ = 0 unavailable or wrong.
    // Measured on the full `gam-solve --lib` suite at that time: 7 failing
    // before, 11 after. The four `rail_face_limit` refusals named the reason
    // exactly —
    //
    //   FaceUnavailable { reason: "the limit fit needed a stabilization ridge
    //   (1.000e-8), so its criterion is not the plain LAML this form expands" }
    //
    // — so the λ→∞ face certificate (#2348) can no longer prove an
    // infinite-smoothing face at all, plus
    // `estimated_nuisance_fits_land_in_the_same_place_cold_and_warm_2363` and
    // `sas_beta_raw_epsilon_sensitivity_matchesfd_at_seed19`. `pls_solver`'s own
    // comment warned about this class from the other direction (#1122: a
    // nonzero δ broke the envelope identity because the derivative was taken on
    // the un-ridged surface while the value used `log|H + δI|`).
    //
    // THE COMPANION FORM NOW CARRIES δ, SO δ IS APPLIED UNCONDITIONALLY.
    //
    // The prerequisite this comment used to name — "carry δ through the
    // companion forms, re-deriving the rail-face λ→∞ expansion with
    // `H = XᵀWX + S_λ + δI`, and only then make δ unconditional" — is done.
    // `LamlFaceParts::stabilization_ridge` carries δ into
    // `laml_rail_face_limit`, which adds it to the same diagonal, so the face
    // form and the criterion expand the same operator. The `rail_face_limit`
    // gate that declined any nonzero δ is gone with it; declining was only
    // ever a way of saying "this form does not know about δ".
    //
    // With that in place, applying δ always is what makes `∂δ/∂ρ = 0` hold
    // identically, which is the invariant `FIXED_STABILIZATION_RIDGE`'s own doc
    // states and which a Cholesky-success predicate breaks.
    //
    // On a well-conditioned Hessian this is numerically inert in the direction
    // that matters: the criterion shifts by `½·Σ ln(1 + δ/λ_i) ≤ ½·δ·tr(H⁻¹)`,
    // far below the convergence tolerances when `λ_i ≫ δ = 1e-8`. What it
    // removes is the 9.21 jump, not the scale.
    // WHAT HAPPENS WHEN THAT IS NOT ENOUGH IS ALSO UNIFORM (#2657). A
    // genuinely non-PD Hessian is a REFUSAL: `eigh` is computed only to report
    // λ_min, and δ stays exactly FIXED_STABILIZATION_RIDGE or there is no fit.
    // The sparse twin follows the same rule, using a Gershgorin lower bound
    // only as diagnostic evidence. Thus `∂δ/∂ρ = 0` holds on every accepted
    // dense and sparse path.
    if ridge > 0.0 {
        for i in 0..hess.nrows() {
            hess[[i, i]] += ridge;
        }
    }
    if hess.cholesky(Side::Lower).is_ok() {
        return Ok(ridge);
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

/// Fold a working model's omitted objective curvature into the LM-regularized
/// Hessian the Newton direction is solved from (#2273).
///
/// The single place the sign convention of
/// [`WorkingModel::objective_hessian_matrix_correction`] is applied, so the
/// direction solve and `objective_hessian_quadratic_correction`'s `-dᵀHΦd`
/// cannot disagree about it. Borrows when there is nothing to fold in, so the
/// ordinary path allocates nothing.
pub(super) fn objective_curvature_for_direction<'h>(
    regularized_hessian: &'h Array2<f64>,
    correction: Option<&Array2<f64>>,
) -> Result<std::borrow::Cow<'h, Array2<f64>>, EstimationError> {
    let Some(correction) = correction else {
        return Ok(std::borrow::Cow::Borrowed(regularized_hessian));
    };
    if correction.dim() != regularized_hessian.dim() {
        crate::bail_invalid_estim!(
            "objective curvature correction shape {}x{} does not match the regularized \
             Hessian {}x{}",
            correction.nrows(),
            correction.ncols(),
            regularized_hessian.nrows(),
            regularized_hessian.ncols()
        );
    }
    let mut curvature = regularized_hessian - correction;
    gam_linalg::matrix::symmetrize_in_place(&mut curvature);
    if !curvature.iter().all(|value| value.is_finite()) {
        crate::bail_invalid_estim!(
            "objective curvature for the Newton direction is non-finite after folding in the \
             model's omitted curvature term"
        );
    }
    Ok(std::borrow::Cow::Owned(curvature))
}

pub(super) fn solve_newton_direction_dense(
    hessian: &Array2<f64>,
    gradient: &Array1<f64>,
    direction_out: &mut Array1<f64>,
) -> Result<(), EstimationError> {
    let dense_solve_start = std::time::Instant::now();
    let p = hessian.nrows();
    if direction_out.len() != gradient.len() {
        *direction_out = Array1::zeros(gradient.len());
    }

    if gam_gpu::cuda_selected()
        .map_err(|error| EstimationError::InvalidInput(error.to_string()))?
    {
        let rhs = Array2::from_shape_vec((p, 1), gradient.to_vec()).map_err(|e| {
            EstimationError::InvalidInput(format!("CUDA PIRLS RHS layout failed: {e}"))
        })?;
        // Solution-only: the Newton direction discards the logdet, so route
        // through the mixed-precision solution-only path that skips the
        // redundant fp64 POTRF (the fp32 factor + fp64 refinement already gives
        // a full-fp64-accurate direction). This is where the mixed-precision
        // speedup is actually realized for the inner Newton solve.
        let solved = crate::gpu::pirls_gpu::cholesky_solve_only_gpu(hessian.view(), rhs.view())
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
            return Ok(());
        }
    }

    let cpu_route = String::from("CPU stable solver");

    let factor = StableSolver::new()
        .factorize(hessian)
        .map_err(EstimationError::LinearSystemSolveFailed)?;
    solve_direction_with_dense_factor(&factor, gradient, direction_out);

    // Validate: bare Cholesky on a near-singular H produces huge spurious
    // step magnitudes in the null direction. If `‖H·δ + g‖∞ / (1+‖g‖∞)` is
    // not small, the purported direction does not solve the requested
    // unperturbed system. Surface that failure to the LM controller rather
    // than silently changing the system or replacing it with a pseudoinverse.
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
        return Err(EstimationError::InvalidInput(format!(
            "PIRLS Newton direction failed its unperturbed linear-system certificate: relative residual {rel:.3e} exceeds 1e-3"
        )));
    }
    if array_is_finite(direction_out) {
        log::info!(
            "[STAGE] PIRLS dense newton solve backend=CPU p={} flops~{} elapsed={:.3}s route=\"{}\"",
            p,
            (p as u64).saturating_mul((p as u64).saturating_mul(p as u64)) / 3,
            dense_solve_start.elapsed().as_secs_f64(),
            cpu_route,
        );
        return Ok(());
    }
    Err(EstimationError::LinearSystemSolveFailed(
        FaerLinalgError::FactorizationFailed {
            context: "PIRLS dense newton solve exhausted",
        },
    ))
}

/// Solve `min_direction ||A direction + residual||` without assembling either
/// `A' A` or the cancellation-prone normal-equation right-hand side
/// `A' residual`. Householder QR sees condition `kappa(A)`, whereas forming
/// the normal equations squares it and can erase stationarity digits when
/// large score and penalty components cancel. When supplied, `firth_hessian`
/// converts the Fisher direction to the exact objective direction through the
/// same QR factor without reconstructing `A' A` or the cancelled score.
pub(super) fn solve_newton_direction_from_root_with_firth_hessian(
    root: &Array2<f64>,
    root_residual: &Array1<f64>,
    firth_hessian: Option<&Array2<f64>>,
    direction_out: &mut Array1<f64>,
) -> Result<f64, EstimationError> {
    let p = root.ncols();
    if root.nrows() < p || root_residual.len() != root.nrows() {
        crate::bail_invalid_estim!(
            "PIRLS square-root solve dimension mismatch: root={}x{}, residual={}",
            root.nrows(),
            p,
            root_residual.len()
        );
    }
    let (q, r) = root
        .qr()
        .map_err(EstimationError::LinearSystemSolveFailed)?;
    if r.nrows() != p || r.ncols() != p {
        crate::bail_invalid_estim!(
            "PIRLS square-root QR produced non-square R={}x{} for p={p}",
            r.nrows(),
            r.ncols()
        );
    }

    // R direction = -Q' residual. Applying Q before triangular substitution
    // preserves the small projected residual directly; forming A' residual
    // first would subtract the large score and penalty gradients in the least
    // accurate coordinate frame.
    let projected_residual = q.t().dot(root_residual);
    if direction_out.len() != p {
        *direction_out = Array1::zeros(p);
    }
    for reverse in 0..p {
        let i = p - 1 - reverse;
        let mut value = -projected_residual[i];
        for k in (i + 1)..p {
            value -= r[[i, k]] * direction_out[k];
        }
        let diagonal = r[[i, i]];
        if !(diagonal.is_finite() && diagonal != 0.0) {
            return Err(EstimationError::ModelIsIllConditioned {
                condition_number: f64::INFINITY,
            });
        }
        direction_out[i] = value / diagonal;
    }

    // Least-squares stationarity certificate evaluated through A itself.
    // `A' (A d + q)` must vanish; its denominator uses the unprojected residual
    // scale, so the test remains meaningful without constructing a rounded
    // Gram or treating a cancelled coefficient-space gradient as input data.
    let mut least_squares_residual = root.dot(direction_out);
    least_squares_residual += root_residual;
    let normal_residual = root.t().dot(&least_squares_residual);
    let residual_inf = inf_norm(normal_residual.iter().copied());
    let root_inf = root
        .rows()
        .into_iter()
        .map(|row| row.iter().map(|value| value.abs()).sum::<f64>())
        .fold(0.0_f64, f64::max);
    let root_transpose_inf = root
        .columns()
        .into_iter()
        .map(|column| column.iter().map(|value| value.abs()).sum::<f64>())
        .fold(0.0_f64, f64::max);
    let direction_inf = inf_norm(direction_out.iter().copied());
    let root_residual_inf = inf_norm(root_residual.iter().copied());
    let scale = root_transpose_inf * (root_inf * direction_inf + root_residual_inf);
    let backward_error = if scale > 0.0 {
        residual_inf / scale
    } else {
        residual_inf
    };
    let tolerance = 256.0 * f64::EPSILON * root.nrows().max(p) as f64;
    if !backward_error.is_finite() || backward_error > tolerance {
        crate::bail_invalid_estim!(
            "PIRLS square-root Newton direction failed its backward-error certificate: \
             error {backward_error:.3e} exceeds {tolerance:.3e}"
        );
    }
    if !array_is_finite(direction_out) {
        crate::bail_invalid_estim!("PIRLS square-root Newton direction is non-finite");
    }
    if let Some(hphi) = firth_hessian {
        let fisher_direction = direction_out.clone();
        let lower = r.t().to_owned();
        correct_fisher_direction_for_firth_hessian_from_root_factor(
            &lower,
            hphi,
            &fisher_direction,
            direction_out,
        )?;
    }
    log::info!(
        "[STAGE] PIRLS dense newton solve backend=CPU p={} rows={} route=\"Householder QR of PSD root\" backward_error={:.3e} damped_decrement_sq={:.3e}",
        p,
        root.nrows(),
        backward_error,
        projected_residual.dot(&projected_residual),
    );
    Ok(projected_residual.dot(&projected_residual))
}

/// Blocked tall-skinny QR for an augmented least-squares root.
///
/// Every full block contains at most `p` original rows. Once a block fills,
/// it is reduced together with the preceding `p × p` triangular factor:
///
/// ```text
/// [R_previous] = Q [R_next]
/// [A_block   ]     [   0  ]
/// ```
///
/// This is algebraically the same Householder QR used by the dense root path,
/// but its peak storage is `O(p²)` instead of `O(rows × p)`. In particular,
/// sparse-native PIRLS designs never become a dense `n × p` allocation merely
/// because a stiff penalty requires the numerically safer square-root solve.
pub(super) struct TallSkinnyQrLeastSquares {
    p: usize,
    pending_root: Array2<f64>,
    pending_residual: Array1<f64>,
    pending_rows: usize,
    triangular_root: Option<Array2<f64>>,
    projected_residual: Array1<f64>,
    total_rows: usize,
    root_row_sum_max: f64,
    root_column_sums: Array1<f64>,
    residual_inf: f64,
}

impl TallSkinnyQrLeastSquares {
    pub(super) fn new(p: usize) -> Result<Self, EstimationError> {
        if p == 0 {
            crate::bail_invalid_estim!("tall-skinny QR requires at least one coefficient");
        }
        Ok(Self {
            p,
            pending_root: Array2::zeros((p, p).f()),
            pending_residual: Array1::zeros(p),
            pending_rows: 0,
            triangular_root: None,
            projected_residual: Array1::zeros(p),
            total_rows: 0,
            root_row_sum_max: 0.0,
            root_column_sums: Array1::zeros(p),
            residual_inf: 0.0,
        })
    }

    pub(super) fn push_row(
        &mut self,
        root_row: ndarray::ArrayView1<'_, f64>,
        root_residual: f64,
    ) -> Result<(), EstimationError> {
        if root_row.len() != self.p {
            crate::bail_invalid_estim!(
                "tall-skinny QR row width {} does not match p={}",
                root_row.len(),
                self.p
            );
        }
        if !root_residual.is_finite() || root_row.iter().any(|value| !value.is_finite()) {
            crate::bail_invalid_estim!("tall-skinny QR received a non-finite augmented row");
        }
        let row_sum = root_row.iter().map(|value| value.abs()).sum::<f64>();
        self.root_row_sum_max = self.root_row_sum_max.max(row_sum);
        for (sum, value) in self.root_column_sums.iter_mut().zip(root_row.iter()) {
            *sum += value.abs();
        }
        self.residual_inf = self.residual_inf.max(root_residual.abs());
        self.pending_root
            .row_mut(self.pending_rows)
            .assign(&root_row);
        self.pending_residual[self.pending_rows] = root_residual;
        self.pending_rows += 1;
        self.total_rows += 1;
        if self.pending_rows == self.p {
            self.flush()?;
        }
        Ok(())
    }

    fn flush(&mut self) -> Result<(), EstimationError> {
        if self.pending_rows == 0 {
            return Ok(());
        }
        let carried_rows = usize::from(self.triangular_root.is_some()) * self.p;
        let rows = carried_rows + self.pending_rows;
        if rows < self.p {
            return Ok(());
        }
        let mut root = Array2::<f64>::zeros((rows, self.p).f());
        let mut residual = Array1::<f64>::zeros(rows);
        if let Some(previous) = self.triangular_root.as_ref() {
            root.slice_mut(ndarray::s![..self.p, ..]).assign(previous);
            residual
                .slice_mut(ndarray::s![..self.p])
                .assign(&self.projected_residual);
        }
        let start = carried_rows;
        let end = start + self.pending_rows;
        root.slice_mut(ndarray::s![start..end, ..])
            .assign(&self.pending_root.slice(ndarray::s![..self.pending_rows, ..]));
        residual
            .slice_mut(ndarray::s![start..end])
            .assign(&self.pending_residual.slice(ndarray::s![..self.pending_rows]));

        let (q, r) = root
            .qr()
            .map_err(EstimationError::LinearSystemSolveFailed)?;
        if r.dim() != (self.p, self.p) {
            crate::bail_invalid_estim!(
                "tall-skinny QR produced R={}x{} for p={}",
                r.nrows(),
                r.ncols(),
                self.p
            );
        }
        self.projected_residual = q.t().dot(&residual);
        self.triangular_root = Some(r);
        self.pending_root.fill(0.0);
        self.pending_residual.fill(0.0);
        self.pending_rows = 0;
        Ok(())
    }

    pub(super) fn solve(
        mut self,
        firth_hessian: Option<&Array2<f64>>,
        direction_out: &mut Array1<f64>,
    ) -> Result<f64, EstimationError> {
        self.flush()?;
        let r = self.triangular_root.ok_or_else(|| {
            EstimationError::InvalidInput(format!(
                "tall-skinny QR has only {} rows for p={}",
                self.total_rows, self.p
            ))
        })?;
        if direction_out.len() != self.p {
            *direction_out = Array1::zeros(self.p);
        }
        for reverse in 0..self.p {
            let i = self.p - 1 - reverse;
            let mut value = -self.projected_residual[i];
            for k in (i + 1)..self.p {
                value -= r[[i, k]] * direction_out[k];
            }
            let diagonal = r[[i, i]];
            if !(diagonal.is_finite() && diagonal != 0.0) {
                return Err(EstimationError::ModelIsIllConditioned {
                    condition_number: f64::INFINITY,
                });
            }
            direction_out[i] = value / diagonal;
        }

        // The block reductions are orthogonal transformations, so the compact
        // R system has exactly the same least-squares stationarity equation as
        // the original augmented root. Certify the triangular solve against
        // norms accumulated from the original rows, not the reduced blocks.
        let mut compact_residual = r.dot(direction_out);
        compact_residual += &self.projected_residual;
        let normal_residual = r.t().dot(&compact_residual);
        let residual_inf = inf_norm(normal_residual.iter().copied());
        let root_transpose_inf = inf_norm(self.root_column_sums.iter().copied());
        let direction_inf = inf_norm(direction_out.iter().copied());
        let scale =
            root_transpose_inf * (self.root_row_sum_max * direction_inf + self.residual_inf);
        let backward_error = if scale > 0.0 {
            residual_inf / scale
        } else {
            residual_inf
        };
        let tolerance = 256.0 * f64::EPSILON * self.total_rows.max(self.p) as f64;
        if !backward_error.is_finite() || backward_error > tolerance {
            crate::bail_invalid_estim!(
                "PIRLS tall-skinny square-root direction failed its backward-error certificate: \
                 error {backward_error:.3e} exceeds {tolerance:.3e}"
            );
        }
        if !array_is_finite(direction_out) {
            crate::bail_invalid_estim!("PIRLS tall-skinny square-root direction is non-finite");
        }
        let decrement = self.projected_residual.dot(&self.projected_residual);
        if let Some(hphi) = firth_hessian {
            let fisher_direction = direction_out.clone();
            let lower = r.t().to_owned();
            correct_fisher_direction_for_firth_hessian_from_root_factor(
                &lower,
                hphi,
                &fisher_direction,
                direction_out,
            )?;
        }
        log::info!(
            "[STAGE] PIRLS tall-skinny newton solve backend=CPU p={} rows={} route=\"blocked Householder QR of sparse PSD root\" backward_error={:.3e} damped_decrement_sq={:.3e}",
            self.p,
            self.total_rows,
            backward_error,
            decrement,
        );
        Ok(decrement)
    }
}

/// Convert the cancellation-safe Fisher/penalty direction into the exact
/// Firth-Newton direction without ever reconstructing the cancelled score.
///
/// Let `H₀ = L Lᵀ` be the damped Fisher-plus-penalty system and let `d₀` solve
/// `H₀ d₀ = -g`, obtained from the augmented QR root.  The exact Firth
/// curvature is `H = H₀ - HΦ`.  Under the Cholesky congruence,
///
/// `C = I - L⁻¹ HΦ L⁻ᵀ`, `C y = Lᵀ d₀`, `d = L⁻ᵀ y`.
///
/// Thus the small cancelled vector `g` is never formed or recovered through
/// `H₀ d₀`; all operations act on the accurately resolved direction `d₀`.
/// Failure of the strict congruence Cholesky means the requested damping does
/// not make the exact objective curvature positive definite, and the LM
/// controller must increase damping rather than silently solve another system.
fn correct_fisher_direction_for_firth_hessian_from_root_factor(
    fisher_root_lower: &Array2<f64>,
    firth_hessian: &Array2<f64>,
    fisher_direction: &Array1<f64>,
    direction_out: &mut Array1<f64>,
) -> Result<(), EstimationError> {
    let p = fisher_root_lower.nrows();
    if fisher_root_lower.ncols() != p
        || firth_hessian.dim() != (p, p)
        || fisher_direction.len() != p
    {
        crate::bail_invalid_estim!(
            "Firth congruence solve dimension mismatch: root={}x{}, Hphi={}x{}, direction={}",
            fisher_root_lower.nrows(),
            fisher_root_lower.ncols(),
            firth_hessian.nrows(),
            firth_hessian.ncols(),
            fisher_direction.len()
        );
    }

    // Left- and right-whiten HΦ.  The right solve is evaluated by transposing:
    // (Y L⁻ᵀ)ᵀ = L⁻¹ Yᵀ.
    let left_whitened = gam_linalg::triangular::forward_substitution_lower_matrix(
        fisher_root_lower,
        firth_hessian,
    );
    let whitened_transpose = gam_linalg::triangular::forward_substitution_lower_matrix(
        fisher_root_lower,
        &left_whitened.t().to_owned(),
    );
    let mut congruence = Array2::<f64>::eye(p) - whitened_transpose.t().to_owned();
    gam_linalg::matrix::symmetrize_in_place(&mut congruence);
    let congruence_factor = congruence
        .cholesky(Side::Lower)
        .map_err(EstimationError::LinearSystemSolveFailed)?;

    let transformed_rhs = fisher_root_lower.t().dot(fisher_direction);
    let transformed_direction = congruence_factor.solvevec(&transformed_rhs);
    let direction = gam_linalg::triangular::back_substitution_lower_transpose(
        fisher_root_lower,
        &transformed_direction,
    );

    let residual = &congruence.dot(&transformed_direction) - &transformed_rhs;
    let residual_inf = inf_norm(residual.iter().copied());
    let congruence_inf = congruence
        .rows()
        .into_iter()
        .map(|row| row.iter().map(|value| value.abs()).sum::<f64>())
        .fold(0.0_f64, f64::max);
    let transformed_direction_inf = inf_norm(transformed_direction.iter().copied());
    let transformed_rhs_inf = inf_norm(transformed_rhs.iter().copied());
    let scale = congruence_inf * transformed_direction_inf + transformed_rhs_inf;
    let backward_error = if scale > 0.0 {
        residual_inf / scale
    } else {
        residual_inf
    };
    let tolerance = 256.0 * f64::EPSILON * p.max(1) as f64;
    if !backward_error.is_finite() || backward_error > tolerance {
        crate::bail_invalid_estim!(
            "Firth congruence Newton direction failed its backward-error certificate: error {backward_error:.3e} exceeds {tolerance:.3e}"
        );
    }
    if !array_is_finite(&direction) {
        crate::bail_invalid_estim!("Firth congruence Newton direction is non-finite");
    }
    if direction_out.len() != p {
        *direction_out = Array1::zeros(p);
    }
    direction_out.assign(&direction);
    Ok(())
}

#[cfg(test)]
mod square_root_solve_tests {
    use super::*;
    use ndarray::array;

    fn solve_newton_direction_from_root(
        root: &Array2<f64>,
        root_residual: &Array1<f64>,
        direction_out: &mut Array1<f64>,
    ) -> Result<f64, EstimationError> {
        solve_newton_direction_from_root_with_firth_hessian(
            root,
            root_residual,
            None,
            direction_out,
        )
    }

    #[test]
    fn qr_root_solve_preserves_a_weak_rotated_direction() {
        // The first two rows have squared-energy ratio 1e10.  Forming A'A
        // subtracts nearly equal O(1e10) entries to recover the weak [1,-1]
        // direction; QR acts on A and keeps that direction directly.
        let root = array![[1.0e5, 1.0e5], [1.0, -1.0], [1.0e-3, 0.0], [0.0, 1.0e-3]];
        let expected = array![1.0, -1.0];
        let residual = -root.dot(&expected);
        let mut actual = Array1::<f64>::zeros(2);

        let decrement = solve_newton_direction_from_root(&root, &residual, &mut actual)
            .expect("square-root solve");

        assert!((actual[0] - expected[0]).abs() < 1.0e-9);
        assert!((actual[1] - expected[1]).abs() < 1.0e-9);
        assert!(decrement.is_finite() && decrement > 0.0);
    }

    #[test]
    fn qr_root_solve_does_not_form_a_cancelling_normal_rhs() {
        let root = array![[1.0e8, 1.0e8], [1.0, -1.0], [0.0, 1.0]];
        let expected = array![0.25, -0.25];
        // This component is exactly orthogonal to both columns of `root` but
        // is much larger than the projected residual in the weak direction.
        let orthogonal_residual = array![1.0e-8, -1.0, -2.0];
        let residual = -root.dot(&expected) + &orthogonal_residual;
        let mut actual = Array1::<f64>::zeros(2);

        solve_newton_direction_from_root(&root, &residual, &mut actual)
            .expect("least-squares root solve");

        assert!((actual[0] - expected[0]).abs() < 1.0e-8);
        assert!((actual[1] - expected[1]).abs() < 1.0e-8);

        // Certification is state-local: at the stationary state the same
        // large root-space residual is orthogonal to the model range, so its
        // bare Newton decrement is machine zero without taking another step
        // and recomputing a cancellation-prone coefficient gradient.
        let mut stationary_direction = Array1::<f64>::zeros(2);
        let stationary_decrement = solve_newton_direction_from_root(
            &root,
            &orthogonal_residual,
            &mut stationary_direction,
        )
        .expect("stationary least-squares root certificate");
        assert!(stationary_direction.dot(&stationary_direction) <= 1.0e-28);
        assert!(stationary_decrement <= 1.0e-28);
    }

    #[test]
    fn tall_skinny_qr_matches_dense_qr_for_a_stiff_root() {
        let root = array![
            [1.0e8, 1.0e8],
            [1.0, -1.0],
            [0.0, 1.0],
            [2.0, 0.0],
            [0.0, 3.0]
        ];
        let expected = array![0.25, -0.25];
        let residual = -root.dot(&expected);
        let mut dense_direction = Array1::<f64>::zeros(2);
        let dense_decrement =
            solve_newton_direction_from_root(&root, &residual, &mut dense_direction)
                .expect("dense square-root solve");

        let mut blocked = TallSkinnyQrLeastSquares::new(2).expect("blocked QR");
        for i in 0..root.nrows() {
            blocked
                .push_row(root.row(i), residual[i])
                .expect("append augmented row");
        }
        let mut blocked_direction = Array1::<f64>::zeros(2);
        let blocked_decrement = blocked
            .solve(None, &mut blocked_direction)
            .expect("blocked square-root solve");

        for (&blocked_value, &dense_value) in
            blocked_direction.iter().zip(dense_direction.iter())
        {
            assert!((blocked_value - dense_value).abs() < 1.0e-10);
        }
        assert!((blocked_decrement - dense_decrement).abs() < 1.0e-8);
    }

    #[test]
    fn firth_congruence_preserves_the_root_direction_without_reforming_the_score() {
        let root = array![[1.0e3, 1.0e3], [1.0, -1.0], [0.0, 1.0]];
        let fisher_hessian = root.t().dot(&root);
        let exact_root_direction = array![0.25, -0.25];
        let root_residual = -root.dot(&exact_root_direction);
        let firth_hessian = array![[0.20, 0.03], [0.03, 0.15]];
        let mut corrected = Array1::<f64>::zeros(2);

        solve_newton_direction_from_root_with_firth_hessian(
            &root,
            &root_residual,
            Some(&firth_hessian),
            &mut corrected,
        )
        .expect("exact Firth congruence solve");

        // Check the congruent equation.  `H0 * d0` is used only in this test
        // oracle; production never reforms this cancellation-prone RHS.
        let true_hessian = &fisher_hessian - &firth_hessian;
        let residual = true_hessian.dot(&corrected) - fisher_hessian.dot(&exact_root_direction);
        assert!(inf_norm(residual.iter().copied()) < 1.0e-8);
        assert!(array_is_finite(&corrected));
    }
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
    op_penalties: &[(f64, &dyn gam_terms::analytic_penalties::PenaltyOp)],
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
        gam_linalg::utils::solve_spd_pcg(apply_h, gradient, &precond_diag, rel_tol, max_iter)
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
pub(crate) const ACTIVE_BOUND_REL_TOL: f64 = 1e-6;

pub(crate) const ACTIVE_BOUND_ABS_TOL: f64 = 1e-10;

pub(super) fn projected_gradient_norm(
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
    // Scale-equivariant objective magnitude for the Δdeviance plateau band.
    //
    // The deviance-change tests below ask "has the penalized objective stopped
    // moving relative to its own magnitude?" — a purely *relative* question.
    // For a Gaussian identity-link fit the deviance is the (weighted) residual
    // sum of squares and the penalty is `βᵀS_λβ`; rescaling the response
    // `y → a·y` rescales `β → a·β` exactly at any fixed λ (the penalized normal
    // equations are linear in `y`), so deviance, penalty, AND the inter-iterate
    // `dev_change` all scale by `a²`. The ratio `dev_change / objective_scale`
    // is therefore scale-invariant, which is exactly what equivariant smoothing
    // selection requires.
    //
    // The previous `.max(1.0)` absolute floor broke this: for a micro-unit
    // response (`a = 1e-6`) the whole objective is `O(a²) ≈ 1e-12`, so the floor
    // pinned the band at `1.0` — ~1e10× too loose — and the inner solve declared
    // a premature plateau at an over-smoothed iterate, which propagated to an
    // inflated `λ̂` (issue #1127). Keying the band to the objective's own
    // magnitude `(|deviance| + |penalty|)` removes the absolute floor while
    // leaving the well-scaled (`a ≳ 1`) and up-scaled (`a = 1e6`) directions
    // byte-identical, since there the floor was already a no-op. When both terms
    // are exactly zero (a perfect interpolating fit) the band is `0`, so the
    // strictly relative `dev_change < 0` test cannot fire spuriously and the
    // separately scale-invariant KKT certificate governs acceptance.
    let objective_scale = state.deviance.abs() + state.penalty_term.abs();
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
            // LM-rejection band: a model-predicted reduction inside the
            // objective's own rounding band `γ_{n+p²}·u·|Φ|` is arithmetic on
            // the quadratic model, not progress (#2469; it replaced a
            // `1e-12·|Φ|` constant). It scales as `O(a²)` under `y → a·y`
            // exactly as the predicted reduction does (#1127). `|Φ|` is a lower
            // bound on the accumulated magnitude when deviance and penalty
            // cancel, which only makes this acceptance rarer, never wider.
            let reduction_noise_floor = super::convergence::objective_rounding_band(
                state.eta.len(),
                state.gradient.len(),
                current_penalized,
            );
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

    // Gradient and objective live on different response scales: for `y → a·y`
    // the projected gradient is `O(a)` while `objective_scale` (deviance +
    // penalty) is `O(a²)`. Compare each against a same-units scale-invariant
    // band — the gradient against the data-driven natural gradient scale via
    // `relative_gradient_norm`, the Δdeviance against `scaled_dev_tol` — so the
    // relative-band plateau is equivariant rather than mixing the two scales
    // (which the old `objective_scale`-only gradient test did, and which the
    // `.max(1.0)` floor then masked at unit scale).
    if state.relative_gradient_norm(projected_grad) <= progress_tol.max(1e-6)
        && dev_change.abs() < scaled_dev_tol * 0.1
        && dev_change >= 0.0
    {
        return Some(PirlsSoftAccept::RelativeBandPlateau);
    }

    None
}

/// The GRADIENT-SPACE stationarity residual of a constrained iterate.
///
/// `gradient`, `beta`, and `linear_constraints` are all represented in the
/// current PIRLS coefficient basis (raw sparse-native or Qs-transformed). At an
/// active inequality the raw gradient can carry a valid KKT multiplier, so
/// convergence must use `‖∇L − Aᵀλ‖∞` in that same frame rather than the
/// unprojected gradient norm.
///
/// # Why this is not the max over all four KKT channels (#2705 group B)
///
/// It used to be
/// `max(primal_feasibility, dual_feasibility, complementarity, stationarity)`,
/// and that scalar was handed to [`WorkingState::certifies_kkt`], whose two
/// bounds — `τ·√n·√p` and `τ·(1 + ‖score‖ + ‖Sβ‖)` — are both derived FOR A
/// GRADIENT: the first from "score components are `O(√n)`", the second from the
/// penalized gradient's own natural magnitude. Only two of the four channels are
/// gradient-space quantities:
///
/// * `stationarity = ‖∇L − Aᵀλ‖∞` and `dual_feasibility = max_i max(0, −λ_i)`
///   both live in gradient units — `λ` is the NNLS projection of the gradient
///   onto the unit-normalized active rows;
/// * `primal_feasibility = max_i max(0, b_i − a_iᵀβ)` is a EUCLIDEAN DISTANCE in
///   COEFFICIENT space (the rows are unit-normalized before it is measured);
/// * `complementarity = max_i |λ_i·s_i|` is a gradient TIMES a distance.
///
/// Measured on `y ~ s(x, shape=convex)`, 300 rows of clean linear data, at the
/// point the fit was refused: `stationarity = 3.148471e-10` against a dimension
/// bound of `6.244998e-9` — twenty times INSIDE the certificate — while
/// `primal_feasibility = 6.301146e-9` pushed the max just past it, by a factor
/// of `1.009`. That feasibility number is itself inside
/// [`crate::active_set::ACTIVE_SET_PRIMAL_FEASIBILITY_TOL`], the tolerance the
/// active-set solver's own docs publish as the one it GUARANTEES on the iterate
/// it returns. The certificate was demanding better feasibility than the solver
/// ever promised, in units the bound was not built for, and refusing a fit whose
/// gradient-space stationarity had already converged.
///
/// The geometric channels keep their obligations — they are certified against
/// their own contracts by [`constraint_geometry_is_certified`], which every
/// acceptance path now requires. This function returns what the gradient
/// certificate is entitled to read.
pub(super) fn constrained_stationarity_norm(
    gradient: &Array1<f64>,
    beta: &Array1<f64>,
    lower_bounds: Option<&Array1<f64>>,
    linear_constraints: Option<&LinearInequalityConstraints>,
) -> f64 {
    if let Some(constraints) = linear_constraints {
        let kkt = compute_constraint_kkt_diagnostics(beta, gradient, constraints);
        return kkt.dual_feasibility.max(kkt.stationarity);
    }
    projected_gradient_norm(gradient, beta, lower_bounds)
}

/// Whether the GEOMETRIC constraint-KKT channels — the two that are not
/// gradient-space quantities — meet the contracts that define them.
///
/// * Primal feasibility is a distance in coefficient space, and the standard it
///   answers to is [`crate::active_set::ACTIVE_SET_PRIMAL_FEASIBILITY_TOL`]:
///   the tolerance the inequality-constrained active-set Newton solver
///   guarantees on the iterate it returns, in the same unit-normalized row
///   metric this residual is measured in. One number, one owner. It is also
///   tighter than the outer startup gate's `KKT_TOL_PRIMAL = 1e-7`, so
///   certifying against it can never hand the outer gate something it rejects.
/// * Complementarity is `|λ_i·s_i|` — a gradient times a distance — and is held
///   to `KKT_TOL_COMP`, the OUTER startup gate's own bound.
///
/// # Why complementarity's bound is inherited rather than derived
///
/// The dimensionally natural bound for `|λ_i·s_i|` scales with the gradient
/// magnitude its multipliers live at; a fixed absolute number makes the same fit
/// pass or fail under a response rescale `y → c·y`, since `λ ∝ c`. But the
/// binding requirement HERE is lockstep: `enforce_constraint_kkt` refuses any
/// iterate whose complementarity exceeds `KKT_TOL_COMP` absolutely, so an inner
/// certificate that admitted the scaled form would certify geometries the outer
/// gate then rejects — and a fit's success would again depend on which ρ the
/// seed loop started from, which is #873. Adopting the natural form is a change
/// to BOTH gates or to neither. This one is recorded as inherited, not endorsed.
///
/// Returns `true` for an unconstrained fit and for a constrained one whose
/// bounds yield no representable constraint rows: there is no geometry to
/// certify, and the gradient certificate carries the whole obligation.
pub(super) fn constraint_geometry_is_certified(
    beta: &Array1<f64>,
    gradient: &Array1<f64>,
    linear_constraints: Option<&LinearInequalityConstraints>,
) -> bool {
    let Some(constraints) = linear_constraints else {
        return true;
    };
    let kkt = compute_constraint_kkt_diagnostics(beta, gradient, constraints);
    kkt.primal_feasibility <= crate::active_set::ACTIVE_SET_PRIMAL_FEASIBILITY_TOL
        && kkt.complementarity <= crate::estimate::reml::outer_eval::KKT_TOL_COMP
}

pub(crate) fn count_dense_upper_nnz(matrix: &Array2<f64>, tol: f64) -> usize {
    let p = matrix.nrows().min(matrix.ncols());
    let mut nnz = 0usize;
    for col in 0..p {
        for row in 0..=col {
            if matrix[[row, col]].abs() > tol {
                nnz += 1;
            }
        }
    }
    nnz
}

pub(crate) fn estimate_sparse_native_decision(
    workspace: &mut PirlsWorkspace,
    x_original: &DesignMatrix,
    s_lambda: &Array2<f64>,
    coefficient_lower_bounds: Option<&Array1<f64>>,
    linear_constraints_original: Option<&LinearInequalityConstraints>,
) -> SparsePirlsDecision {
    let p = x_original.ncols();
    let nnz_s_lambda = count_dense_upper_nnz(s_lambda, 1e-12);
    let dense_reject = |reason: &'static str, nnz_x: usize| SparsePirlsDecision {
        path: PirlsLinearSolvePath::DenseTransformed,
        reason,
        p,
        nnz_x,
        nnz_xtwx_symbolic: None,
        nnz_s_lambda,
        nnz_h_est: None,
        density_h_est: None,
    };

    // Constrained solves require the dense active-set / projected Newton machinery.
    let has_finite_lower_bounds = coefficient_lower_bounds
        .map(|lb| lb.iter().any(|bound| bound.is_finite()))
        .unwrap_or(false);
    if has_finite_lower_bounds || linear_constraints_original.is_some() {
        return dense_reject("constraints_present", 0);
    }

    let x_sparse = if let Some(sparse) = x_original.as_sparse() {
        sparse
    } else {
        // Count nonzeros via chunks so operator-backed dense designs
        // (e.g. lazy ScaleDeviationOperator) participate in this diagnostic
        // path without forcing a full materialization.
        let row_chunk_start = std::time::Instant::now();
        let n = x_original.nrows();
        let chunk = row_chunk_for_byte_budget(n, x_original.ncols());
        let mut nnz: usize = 0;
        let mut chunks_processed = 0usize;
        if chunk > 0 && n > 0 {
            let mut start = 0;
            while start < n {
                let end = (start + chunk).min(n);
                chunks_processed += 1;
                match x_original.try_row_chunk(start..end) {
                    Ok(rows) => {
                        nnz = nnz.saturating_add(rows.iter().filter(|v| v.abs() > 1e-12).count());
                    }
                    Err(_) => {
                        nnz = nnz.saturating_add((end - start).saturating_mul(x_original.ncols()));
                    }
                }
                start = end;
            }
        }
        log::info!(
            "[STAGE] PIRLS row-chunk generation chunks={} n={} p={} nnz={} elapsed={:.3}s",
            chunks_processed,
            n,
            x_original.ncols(),
            nnz,
            row_chunk_start.elapsed().as_secs_f64(),
        );
        return dense_reject("design_not_sparse", nnz);
    };
    let nnz_x = x_sparse.val().len();
    match workspace.sparse_penalized_system_stats(x_sparse, s_lambda) {
        Ok(stats) => SparsePirlsDecision {
            path: if stats.density_upper <= SPARSE_NATIVE_MAX_H_DENSITY {
                PirlsLinearSolvePath::SparseNative
            } else {
                PirlsLinearSolvePath::DenseTransformed
            },
            reason: if stats.density_upper <= SPARSE_NATIVE_MAX_H_DENSITY {
                "sparse_native_eligible"
            } else {
                "penalized_hessian_too_dense"
            },
            p,
            nnz_x,
            nnz_xtwx_symbolic: Some(stats.nnz_xtwx_symbolic),
            nnz_s_lambda: stats.nnz_s_lambda_upper,
            nnz_h_est: Some(stats.nnz_h_upper),
            density_h_est: Some(stats.density_upper),
        },
        Err(_) => dense_reject("sparse_stats_failed", nnz_x),
    }
}

pub(super) fn should_use_sparse_native_pirls(
    workspace: &mut PirlsWorkspace,
    x_original: &DesignMatrix,
    s_lambda: &Array2<f64>,
    coefficient_lower_bounds: Option<&Array1<f64>>,
    linear_constraints_original: Option<&LinearInequalityConstraints>,
) -> SparsePirlsDecision {
    estimate_sparse_native_decision(
        workspace,
        x_original,
        s_lambda,
        coefficient_lower_bounds,
        linear_constraints_original,
    )
}

/// Assemble a sparse SPD Hessian with the fixed stabilization ridge.
///
/// The returned matrix and factor always carry exactly
/// [`FIXED_STABILIZATION_RIDGE`]. If that matrix cannot be factorized, the
/// helper refuses rather than selecting a larger shift from the Hessian.
/// Choosing a shift from `H(ρ)` would make the outer criterion's
/// `0.5·log|H(ρ) + δI|` use `δ(ρ)` while its envelope derivative omits
/// `∂δ/∂ρ` (#2657).
///
/// Returning the factor avoids the previous double-factorization where the SPD
/// check would factor the matrix and discard the factor, then the caller would
/// immediately call `factorize_sparse_spd` again on the same matrix to solve.
pub(super) fn ensure_sparse_positive_definite_with_fixed_ridge<F>(
    mut assemble: F,
) -> Result<
    (
        SparseColMat<usize, f64>,
        gam_linalg::sparse_exact::SparseExactFactor,
        f64,
    ),
    EstimationError,
>
where
    F: FnMut(f64) -> Result<SparseColMat<usize, f64>, EstimationError>,
{
    // Step 1 — the fixed stabilization ridge, applied UNCONDITIONALLY.
    //
    // A symmetric Hessian assembled from `XᵀWX + S_λ` is mathematically PSD;
    // the only reason an exact-arithmetic PSD matrix fails a Cholesky is
    // floating-point round-off in the assembly, which a fixed tiny nugget on
    // the diagonal cures. This is the principled, scale-free first attempt and
    // the common case.
    //
    // δ IS NOT CHOSEN BY A BRANCH. This ladder used to try `assemble(0.0)`
    // first and return `ridge = 0.0` when that factorized. Two things were
    // wrong with that:
    //
    //  1. A Cholesky-success predicate on a near-singular matrix is a function
    //     of ρ, so δ became a function of ρ — and δ enters the outer criterion
    //     through `0.5·log|H|`. `FIXED_STABILIZATION_RIDGE`'s own doc in
    //     `gam_working_model.rs` states the invariant: δ must be constant
    //     w.r.t. ρ or the envelope-theorem gradient `dV/dρ_k` is invalid. The
    //     dense twin (`ensure_positive_definitewithridge`) was measured jumping
    //     by exactly `0.5·ln(1e8) = 9.21` between neighbouring ρ for this
    //     reason (#1575/#2519), which is what kills the outer line search.
    //
    //  2. It reported the REQUESTED ridge, not the APPLIED one. `pls_solver`'s
    //     sparse branch passed a closure that rewrote a requested `0.0` into
    //     `FIXED_STABILIZATION_RIDGE`, so the first rung returned a matrix
    //     carrying δ = 1e-8 together with `ridge_used = 0.0`. β̂ was then the
    //     stationary point of the RIDGED system while the criterion was
    //     assembled as if unridged: the Tikhonov RHS term `δ·μ` was skipped,
    //     `penalty_term += δ‖β‖²` was skipped, and `ridge_passport.delta()`
    //     reported 0 to every consumer. Asking for δ up front makes the
    //     reported ridge equal the applied ridge by construction.
    //
    // A secondary consequence of the old order: when the first factorization
    // failed, `assemble(FIXED_STABILIZATION_RIDGE)` produced a BIT-IDENTICAL
    // matrix under that clamping closure, so it failed again and control fell
    // through to the former Gershgorin escalation, which set a DATA-DEPENDENT
    // τ(ρ) — an unbounded ρ-dependent jump in `log|H|`, strictly worse than the
    // 9.21 one. That escalation is gone: the bound below diagnoses the refusal
    // but cannot alter δ.
    let h_eps = assemble(FIXED_STABILIZATION_RIDGE)?;
    if let Ok(factor) = factorize_sparse_spd(&h_eps) {
        return Ok((h_eps, factor, FIXED_STABILIZATION_RIDGE));
    }

    // The bound is diagnostic only. It may depend on H(ρ), but it never changes
    // the accepted matrix or objective.
    let gershgorin_min = gershgorin_min_eigenvalue_lower_bound(&h_eps);

    Err(EstimationError::HessianNotPositiveDefinite {
        min_eigenvalue: gershgorin_min,
    })
}

/// Rigorous lower bound on the smallest eigenvalue of a symmetric sparse matrix
/// via Gershgorin's circle theorem.
///
/// The bound is storage-agnostic: off-diagonal magnitudes are added to the
/// radius of both endpoints, so upper-only, lower-only, and full-symmetric
/// storage all yield a valid (and at worst conservative) lower bound — it never
/// over-claims positive-definiteness.
pub(crate) fn gershgorin_min_eigenvalue_lower_bound(h: &SparseColMat<usize, f64>) -> f64 {
    let n = h.ncols();
    let mut diag = vec![0.0_f64; n];
    let mut radius = vec![0.0_f64; n];
    let (symbolic, values) = h.parts();
    let col_ptr = symbolic.col_ptr();
    let row_idx = symbolic.row_idx();
    for col in 0..n {
        let start = col_ptr[col];
        let end = col_ptr[col + 1];
        for idx in start..end {
            let row = row_idx[idx];
            let value = values[idx];
            if row == col {
                diag[col] += value;
            } else {
                let a = value.abs();
                radius[row] += a;
                radius[col] += a;
            }
        }
    }
    let mut min_bound = f64::INFINITY;
    for i in 0..n {
        min_bound = min_bound.min(diag[i] - radius[i]);
    }
    if !min_bound.is_finite() {
        min_bound = f64::NEG_INFINITY;
    }
    min_bound
}

pub(crate) fn solve_subsystem_direction(
    h_sub: ndarray::ArrayView2<f64>,
    g_sub: ndarray::ArrayView1<f64>,
    out: &mut Array1<f64>,
) -> Result<(), EstimationError> {
    let n = g_sub.len();
    if out.len() != n {
        *out = Array1::zeros(n);
    }
    let factor = StableSolver::new()
        .factorize_any(&h_sub)
        .map_err(EstimationError::LinearSystemSolveFailed)?;
    out.assign(&g_sub);
    let mut rhs = array1_to_col_matmut(out);
    factor.solve_in_place(rhs.as_mut());
    out.mapv_inplace(|value| -value);
    if array_is_finite(out) {
        Ok(())
    } else {
        Err(EstimationError::InvalidInput(
            "PIRLS constrained subsystem solve produced a non-finite direction".to_string(),
        ))
    }
}

pub(super) fn linear_constraints_from_lower_bounds(
    lower_bounds: &Array1<f64>,
) -> Option<LinearInequalityConstraints> {
    LinearInequalityConstraints::from_per_coordinate_lower_bounds(lower_bounds)
}

pub(super) fn compute_constraint_kkt_diagnostics(
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
pub(super) fn select_active_set_release(
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
            // Scale-aware deadband (identical to Bland's branch above). A
            // multiplier that is negative only at round-off level is KKT-feasible
            // and MUST NOT trigger a release: releasing an essentially-tight bound
            // on floating-point noise lets the freed coefficient step away, only
            // for the bound to be re-added on the next outer re-linearization —
            // the classic active-set zigzag (gam#979). A genuinely-negative
            // multiplier (below `-tol`) still releases, so this is a strict
            // no-op at any true constrained optimum where multipliers are >= 0.
            let tol = 64.0 * f64::EPSILON * gradient[i].abs().max(hd[i].abs()).max(1.0);
            if lambda_i < -tol && lambda_i < worst {
                worst = lambda_i;
                idx = Some(i);
            }
        }
        idx
    }
}

pub fn solve_newton_directionwith_lower_bounds(
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

        // Dual feasibility belongs to the same quadratic model as the free
        // solve. Using a different Hessian (or dropping `H d`) makes release and
        // entry mutually inconsistent and can cycle forever (gam#979).
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

    Err(EstimationError::InvalidInput(format!(
        "lower-bound active-set QP did not reach a consistent primal/dual KKT set in {max_iters} pivots"
    )))
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
pub(super) fn solve_newton_directionwith_linear_constraints(
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
