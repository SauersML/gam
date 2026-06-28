use crate::LinalgError;
use crate::faer_ndarray::{FaerArrayView, FaerColView};
use faer::Side;
use faer::linalg::solvers::Solve;
use faer::sparse::linalg::solvers::Llt as SparseLlt;
use faer::sparse::{SparseColMat, SymbolicSparseColMat, Triplet};
use ndarray::{Array1, Array2, ArrayBase, ArrayView2, Data, Ix1, Ix2};
use rayon::prelude::*;
use std::collections::BTreeMap;
use std::sync::{Arc, Mutex};

const ZERO_TOL: f64 = 1e-12;
const PARALLEL_SPARSE_FILL_COLUMN_THRESHOLD: usize = 64;

macro_rules! bail_invalid_linalg {
    ($($arg:tt)*) => {
        return Err(LinalgError::InvalidInput(format!($($arg)*)))
    };
}

#[derive(Clone)]
pub struct SparseExactFactor {
    factor: SparseLlt<usize, f64>,
    simplicial: Arc<SimplicialFactor>,
    n: usize,
    logdet: f64,
}

impl crate::matrix::FactorizedSystem for SparseExactFactor {
    fn solve(&self, rhs: &Array1<f64>) -> Result<Array1<f64>, String> {
        solve_sparse_spd(self, rhs).map_err(|e| e.to_string())
    }

    fn solvemulti(&self, rhs: &Array2<f64>) -> Result<Array2<f64>, String> {
        solve_sparse_spdmulti(self, rhs).map_err(|e| e.to_string())
    }

    fn logdet(&self) -> f64 {
        self.logdet
    }
}

pub fn dense_to_sparse(
    matrix: &Array2<f64>,
    tol: f64,
) -> Result<SparseColMat<usize, f64>, LinalgError> {
    let nrows = matrix.nrows();
    let ncols = matrix.ncols();
    // Direct column-major CSC construction.  Three-pass: count nnz per
    // column in parallel, perform the prefix sum serially, then fill each
    // deterministic column slice in parallel.  Columns are still traversed
    // in order and rows are written in ascending order within each column,
    // preserving the same canonical CSC ordering as the previous serial
    // implementation without requiring a triplet sort/dedup pass.
    let counts: Vec<usize> = (0..ncols)
        .into_par_iter()
        .map(|col| {
            let mut count = 0usize;
            for row in 0..nrows {
                if matrix[[row, col]].abs() > tol {
                    count += 1;
                }
            }
            count
        })
        .collect();
    let col_ptr = prefix_sum_counts(&counts);
    let nnz = col_ptr[ncols];
    let mut row_idx = vec![0usize; nnz];
    let mut values = vec![0.0; nnz];
    fill_dense_to_sparse_columns(matrix, tol, 0, ncols, &col_ptr, &mut row_idx, &mut values);
    let symbolic = SymbolicSparseColMat::<usize>::new_checked(nrows, ncols, col_ptr, None, row_idx);
    Ok(SparseColMat::<usize, f64>::new(symbolic, values))
}

/// Convert a dense symmetric matrix to sparse CSC storing only the upper triangle.
///
/// This encoding is required by sparse SPD routines in this module that interpret
/// entries as symmetric-upper storage and mirror off-diagonals when reconstructing
/// dense diagnostics.
pub fn dense_to_sparse_symmetric_upper(
    matrix: &Array2<f64>,
    tol: f64,
) -> Result<SparseColMat<usize, f64>, LinalgError> {
    let nrows = matrix.nrows();
    let ncols = matrix.ncols();
    // Direct CSC build over the upper triangle.  Counts and fills are
    // parallelized by column, with a serial prefix sum between them so every
    // column writes to a deterministic, non-overlapping slice.  Iterating rows
    // from low to high within each column keeps CSC row indices sorted exactly
    // as in the previous serial implementation.
    let row_limit = nrows.min(ncols);
    let counts: Vec<usize> = (0..ncols)
        .into_par_iter()
        .map(|col| {
            let mut count = 0usize;
            let row_end = (col + 1).min(row_limit);
            for row in 0..row_end {
                if matrix[[row, col]].abs() > tol {
                    count += 1;
                }
            }
            count
        })
        .collect();
    let col_ptr = prefix_sum_counts(&counts);
    let nnz = col_ptr[ncols];
    let mut row_idx = vec![0usize; nnz];
    let mut values = vec![0.0; nnz];
    fill_dense_symmetric_upper_columns(
        matrix,
        tol,
        row_limit,
        0,
        ncols,
        &col_ptr,
        &mut row_idx,
        &mut values,
    );
    let symbolic = SymbolicSparseColMat::<usize>::new_checked(nrows, ncols, col_ptr, None, row_idx);
    Ok(SparseColMat::<usize, f64>::new(symbolic, values))
}

fn prefix_sum_counts(counts: &[usize]) -> Vec<usize> {
    let mut col_ptr = Vec::with_capacity(counts.len() + 1);
    col_ptr.push(0);
    let mut running = 0usize;
    for &count in counts {
        running += count;
        col_ptr.push(running);
    }
    col_ptr
}

fn fill_dense_to_sparse_columns(
    matrix: &Array2<f64>,
    tol: f64,
    col_start: usize,
    col_end: usize,
    col_ptr: &[usize],
    row_idx: &mut [usize],
    values: &mut [f64],
) {
    if col_end - col_start <= PARALLEL_SPARSE_FILL_COLUMN_THRESHOLD {
        let base = col_ptr[col_start];
        for col in col_start..col_end {
            let mut write = col_ptr[col] - base;
            for row in 0..matrix.nrows() {
                let value = matrix[[row, col]];
                if value.abs() > tol {
                    row_idx[write] = row;
                    values[write] = value;
                    write += 1;
                }
            }
        }
        return;
    }

    let mid = col_start + (col_end - col_start) / 2;
    let split = col_ptr[mid] - col_ptr[col_start];
    let (left_rows, right_rows) = row_idx.split_at_mut(split);
    let (left_values, right_values) = values.split_at_mut(split);
    rayon::join(
        || {
            fill_dense_to_sparse_columns(
                matrix,
                tol,
                col_start,
                mid,
                col_ptr,
                left_rows,
                left_values,
            );
        },
        || {
            fill_dense_to_sparse_columns(
                matrix,
                tol,
                mid,
                col_end,
                col_ptr,
                right_rows,
                right_values,
            );
        },
    );
}

fn fill_dense_symmetric_upper_columns(
    matrix: &Array2<f64>,
    tol: f64,
    row_limit: usize,
    col_start: usize,
    col_end: usize,
    col_ptr: &[usize],
    row_idx: &mut [usize],
    values: &mut [f64],
) {
    if col_end - col_start <= PARALLEL_SPARSE_FILL_COLUMN_THRESHOLD {
        let base = col_ptr[col_start];
        for col in col_start..col_end {
            let row_end = (col + 1).min(row_limit);
            let mut write = col_ptr[col] - base;
            for row in 0..row_end {
                let value = matrix[[row, col]];
                if value.abs() > tol {
                    row_idx[write] = row;
                    values[write] = value;
                    write += 1;
                }
            }
        }
        return;
    }

    let mid = col_start + (col_end - col_start) / 2;
    let split = col_ptr[mid] - col_ptr[col_start];
    let (left_rows, right_rows) = row_idx.split_at_mut(split);
    let (left_values, right_values) = values.split_at_mut(split);
    rayon::join(
        || {
            fill_dense_symmetric_upper_columns(
                matrix,
                tol,
                row_limit,
                col_start,
                mid,
                col_ptr,
                left_rows,
                left_values,
            );
        },
        || {
            fill_dense_symmetric_upper_columns(
                matrix,
                tol,
                row_limit,
                mid,
                col_end,
                col_ptr,
                right_rows,
                right_values,
            );
        },
    );
}

pub fn sparse_symmetric_upper_matvec_public<S: Data<Elem = f64>>(
    matrix: &SparseColMat<usize, f64>,
    vector: &ArrayBase<S, Ix1>,
) -> Array1<f64> {
    let mut out = Array1::<f64>::zeros(matrix.nrows());
    let (symbolic, values) = matrix.parts();
    let col_ptr = symbolic.col_ptr();
    let row_idx = symbolic.row_idx();
    for col in 0..matrix.ncols() {
        let x_col = vector[col];
        for idx in col_ptr[col]..col_ptr[col + 1] {
            let row = row_idx[idx];
            let value = values[idx];
            out[row] += value * x_col;
            if row != col {
                out[col] += value * vector[row];
            }
        }
    }
    out
}

pub fn factorize_sparse_spd(
    h: &SparseColMat<usize, f64>,
) -> Result<SparseExactFactor, LinalgError> {
    // Canonicalize to symmetric-upper storage before factorization.
    //
    // Math contract:
    // - If callers pass upper-only storage, values are preserved.
    // - If callers pass full symmetric storage, paired (i,j)/(j,i) entries are averaged.
    // - If callers pass lower-only storage, it is mirrored into upper.
    //
    // This prevents off-diagonal double counting in paths that interpret input as
    // symmetric-upper and makes the sparse factor path robust to caller encoding.
    let t_start = std::time::Instant::now();
    let n_input = h.ncols();
    let h_upper = canonicalize_sparse_symmetric_upper(h, ZERO_TOL)?;
    let factor = h_upper.as_ref().sp_cholesky(Side::Upper).map_err(|_| {
        LinalgError::ModelIsIllConditioned {
            condition_number: f64::INFINITY,
        }
    })?;
    // Keep an explicit simplicial LLᵀ factor in addition to faer's solver
    // object. The raw L is needed by callers that must reconstruct H in a
    // changed basis, such as active-constraint tangent projection.
    let simplicial = factorize_simplicial_canonical_upper(&h_upper)?;
    let logdet = simplicial.logdet;
    let elapsed_ms = t_start.elapsed().as_secs_f64() * 1000.0;
    if elapsed_ms > 100.0 {
        log::info!(
            "[sparse-chol] factorize_sparse_spd | n={} | {:.1}ms",
            n_input,
            elapsed_ms
        );
    }
    Ok(SparseExactFactor {
        factor,
        simplicial: Arc::new(simplicial),
        n: h_upper.ncols(),
        logdet,
    })
}

fn canonicalize_sparse_symmetric_upper(
    matrix: &SparseColMat<usize, f64>,
    tol: f64,
) -> Result<SparseColMat<usize, f64>, LinalgError> {
    if matrix.nrows() != matrix.ncols() {
        bail_invalid_linalg!(
            "sparse SPD factorization requires square matrix, got {}x{}",
            matrix.nrows(),
            matrix.ncols()
        );
    }

    #[derive(Default, Clone, Copy)]
    struct PairAccum {
        upper_sum: f64,
        upper_count: usize,
        lower_sum: f64,
        lower_count: usize,
    }

    let mut accum: BTreeMap<(usize, usize), PairAccum> = BTreeMap::new();
    let (symbolic, values) = matrix.parts();
    let col_ptr = symbolic.col_ptr();
    let row_idx = symbolic.row_idx();

    for col in 0..matrix.ncols() {
        let start = col_ptr[col];
        let end = col_ptr[col + 1];
        for idx in start..end {
            let row = row_idx[idx];
            let value = values[idx];
            let (r, c, is_upper) = if row <= col {
                (row, col, true)
            } else {
                (col, row, false)
            };
            let slot = accum.entry((r, c)).or_default();
            if is_upper {
                slot.upper_sum += value;
                slot.upper_count += 1;
            } else {
                slot.lower_sum += value;
                slot.lower_count += 1;
            }
        }
    }

    let mut triplets = Vec::<Triplet<usize, usize, f64>>::new();
    for ((row, col), slot) in accum {
        let value = if row == col {
            let count = slot.upper_count + slot.lower_count;
            if count == 0 {
                0.0
            } else {
                (slot.upper_sum + slot.lower_sum) / (count as f64)
            }
        } else {
            let upper_avg = if slot.upper_count > 0 {
                Some(slot.upper_sum / (slot.upper_count as f64))
            } else {
                None
            };
            let lower_avg = if slot.lower_count > 0 {
                Some(slot.lower_sum / (slot.lower_count as f64))
            } else {
                None
            };
            match (upper_avg, lower_avg) {
                (Some(u), Some(l)) => 0.5 * (u + l),
                (Some(u), None) => u,
                (None, Some(l)) => l,
                (None, None) => 0.0,
            }
        };

        if value.abs() > tol {
            triplets.push(Triplet::new(row, col, value));
        }
    }

    SparseColMat::try_new_from_triplets(matrix.nrows(), matrix.ncols(), &triplets).map_err(|_| {
        LinalgError::InvalidInput(
            "failed to canonicalize sparse matrix to symmetric-upper CSC".to_string(),
        )
    })
}

fn solve_view<R, I, F>(
    factor: &SparseExactFactor,
    rhs: ArrayView2<'_, f64>,
    indices: I,
    mut result: R,
    non_finite_message: &'static str,
    mut consume: F,
) -> Result<R, LinalgError>
where
    I: IntoIterator<Item = (usize, usize)>,
    F: FnMut(&mut R, usize, usize, f64),
{
    let rhsview = FaerArrayView::new(&rhs);
    let solved = factor.factor.solve(rhsview.as_ref());
    for (row, col) in indices {
        let value = solved[(row, col)];
        if !value.is_finite() {
            bail_invalid_linalg!("{}", non_finite_message.to_string());
        }
        consume(&mut result, row, col, value);
    }
    Ok(result)
}

pub fn solve_sparse_spd<S>(
    factor: &SparseExactFactor,
    rhs: &ArrayBase<S, Ix1>,
) -> Result<Array1<f64>, LinalgError>
where
    S: Data<Elem = f64>,
{
    if rhs.len() != factor.n {
        bail_invalid_linalg!(
            "sparse SPD solve dimension mismatch: rhs has {}, factor has {}",
            rhs.len(),
            factor.n
        );
    }
    let mut result = Array1::<f64>::zeros(rhs.len());
    solve_sparse_spd_into(factor, rhs, &mut result)?;
    Ok(result)
}

/// In-place variant of [`solve_sparse_spd`]. Writes the solution directly into
/// `out`, avoiding the intermediate `Array1` allocation on the hot PIRLS path.
/// `out` must already be sized to match `factor.n` (typically the reused
/// Newton-direction buffer).
pub fn solve_sparse_spd_into<S>(
    factor: &SparseExactFactor,
    rhs: &ArrayBase<S, Ix1>,
    out: &mut Array1<f64>,
) -> Result<(), LinalgError>
where
    S: Data<Elem = f64>,
{
    if rhs.len() != factor.n {
        bail_invalid_linalg!(
            "sparse SPD solve dimension mismatch: rhs has {}, factor has {}",
            rhs.len(),
            factor.n
        );
    }
    if out.len() != factor.n {
        bail_invalid_linalg!(
            "sparse SPD solve output dimension mismatch: out has {}, factor has {}",
            out.len(),
            factor.n
        );
    }
    let rhsview = FaerColView::new(rhs);
    let solved = factor.factor.solve(rhsview.as_ref());
    for i in 0..factor.n {
        let value = solved[(i, 0)];
        if !value.is_finite() {
            bail_invalid_linalg!("sparse SPD solve produced non-finite values");
        }
        out[i] = value;
    }
    Ok(())
}

pub fn solve_sparse_spdmulti<S>(
    factor: &SparseExactFactor,
    rhs: &ArrayBase<S, Ix2>,
) -> Result<Array2<f64>, LinalgError>
where
    S: Data<Elem = f64>,
{
    if rhs.nrows() != factor.n {
        bail_invalid_linalg!(
            "sparse SPD multi-solve row mismatch: rhs has {}, factor has {}",
            rhs.nrows(),
            factor.n
        );
    }
    let indices = (0..rhs.nrows()).flat_map(|i| (0..rhs.ncols()).map(move |j| (i, j)));
    solve_view(
        factor,
        rhs.view(),
        indices,
        Array2::<f64>::zeros(rhs.raw_dim()),
        "sparse SPD multi-solve produced non-finite values",
        |result, row, col, value| {
            result[[row, col]] = value;
        },
    )
}

pub fn solve_sparse_spdmulti_rows<S>(
    factor: &SparseExactFactor,
    rhs: &ArrayBase<S, Ix2>,
    row_start: usize,
    row_end: usize,
) -> Result<Array2<f64>, LinalgError>
where
    S: Data<Elem = f64>,
{
    if rhs.nrows() != factor.n {
        bail_invalid_linalg!(
            "sparse SPD multi-solve row mismatch: rhs has {}, factor has {}",
            rhs.nrows(),
            factor.n
        );
    }
    if row_start > row_end || row_end > factor.n {
        bail_invalid_linalg!(
            "sparse SPD selected rows out of bounds: row_start={}, row_end={}, factor={}",
            row_start,
            row_end,
            factor.n
        );
    }
    let indices = (row_start..row_end).flat_map(|i| (0..rhs.ncols()).map(move |j| (i, j)));
    solve_view(
        factor,
        rhs.view(),
        indices,
        Array2::<f64>::zeros((row_end - row_start, rhs.ncols())),
        "sparse SPD selected-row solve produced non-finite values",
        |result, row, col, value| {
            result[[row - row_start, col]] = value;
        },
    )
}

pub fn solve_sparse_spdmulti_diagonal_sum<S>(
    factor: &SparseExactFactor,
    rhs: &ArrayBase<S, Ix2>,
    row_start: usize,
) -> Result<f64, LinalgError>
where
    S: Data<Elem = f64>,
{
    if row_start.saturating_add(rhs.ncols()) > rhs.nrows() {
        bail_invalid_linalg!(
            "sparse SPD selected diagonal out of bounds: row_start={}, rows={}, cols={}",
            row_start,
            rhs.nrows(),
            rhs.ncols()
        );
    }
    let indices = (0..rhs.ncols()).map(|col| (row_start + col, col));
    solve_view(
        factor,
        rhs.view(),
        indices,
        0.0,
        "sparse SPD selected diagonal solve produced non-finite values",
        |sum, _, _, value| {
            *sum += value;
        },
    )
}

pub fn logdet_from_factor(factor: &SparseExactFactor) -> Result<f64, LinalgError> {
    Ok(factor.logdet)
}

pub fn assemble_sparse_factor_h_dense(
    factor: &SparseExactFactor,
) -> Result<Array2<f64>, LinalgError> {
    factor.simplicial.assemble_h_dense_original_order()
}

// ---------------------------------------------------------------------------
// Takahashi selected inversion via simplicial Cholesky
// ---------------------------------------------------------------------------

use faer::dyn_stack::{MemBuffer, MemStack, StackReq};
use faer::linalg::cholesky::llt::factor::LltRegularization;
use faer::sparse::linalg::amd;
use faer::sparse::linalg::cholesky::simplicial;

/// A simplicial Cholesky factorization with raw access to L's CSC pattern and
/// values, plus the AMD permutation.  Built using faer's low-level simplicial
/// API so that L's sparse structure is directly available for Takahashi
/// selected inversion.
pub struct SimplicialFactor {
    /// Column pointers of L (lower triangular, CSC), length n+1
    l_col_ptr: Vec<usize>,
    /// Row indices of L (lower triangular, CSC), length nnz(L)
    l_row_idx: Vec<usize>,
    /// Numeric values of L, length nnz(L)
    l_values: Vec<f64>,
    /// Inverse permutation returned by faer, used to map original coordinates
    /// into the permuted simplicial factor basis.
    perm_inv: Vec<usize>,
    /// Dimension
    n: usize,
    /// log|H| = 2 * sum(log(L_ii))
    pub logdet: f64,
}

/// Build a [`SimplicialFactor`] from a symmetric CSC matrix (upper, lower, or
/// full storage – it is canonicalized to symmetric-upper internally).
///
/// The factorization uses AMD fill-reducing ordering and faer's simplicial
/// LLᵀ numeric factorization.
pub fn factorize_simplicial(h: &SparseColMat<usize, f64>) -> Result<SimplicialFactor, LinalgError> {
    let h_upper = canonicalize_sparse_symmetric_upper(h, ZERO_TOL)?;
    factorize_simplicial_canonical_upper(&h_upper)
}

fn factorize_simplicial_canonical_upper(
    h_upper: &SparseColMat<usize, f64>,
) -> Result<SimplicialFactor, LinalgError> {
    let n = h_upper.ncols();
    if n == 0 {
        return Ok(SimplicialFactor {
            l_col_ptr: vec![0],
            l_row_idx: Vec::new(),
            l_values: Vec::new(),
            perm_inv: Vec::new(),
            n: 0,
            logdet: 0.0,
        });
    }

    let a_nnz = h_upper.compute_nnz();

    // 1. AMD ordering
    let mut perm_fwd = vec![0usize; n];
    let mut perm_inv = vec![0usize; n];
    {
        let mut mem = MemBuffer::new(amd::order_scratch::<usize>(n, a_nnz));
        amd::order(
            &mut perm_fwd,
            &mut perm_inv,
            h_upper.symbolic(),
            amd::Control::default(),
            MemStack::new(&mut mem),
        )
        .map_err(|_| LinalgError::ModelIsIllConditioned {
            condition_number: f64::INFINITY,
        })?;
    }

    // perm_fwd and perm_inv have length n and were just populated by
    // amd::order above for a valid symmetric n×n CSC matrix. On success,
    // amd::order writes a valid permutation of 0..n into perm_fwd and its
    // exact inverse into perm_inv.
    // SAFETY: those are exactly the invariants required by PermRef::new_unchecked.
    let perm = unsafe { faer::perm::PermRef::new_unchecked(&perm_fwd, &perm_inv, n) };

    // 2. Permute to P A Pᵀ (upper-triangular, unsorted)
    let a_perm_upper = {
        let mut col_ptrs = vec![0usize; n + 1];
        let mut row_indices = vec![0usize; a_nnz];
        let mut values = vec![0.0f64; a_nnz];
        let mut mem = MemBuffer::new(faer::sparse::utils::permute_self_adjoint_scratch::<usize>(
            n,
        ));
        faer::sparse::utils::permute_self_adjoint_to_unsorted(
            &mut values,
            &mut col_ptrs,
            &mut row_indices,
            h_upper.as_ref(),
            perm,
            Side::Upper,
            Side::Upper,
            MemStack::new(&mut mem),
        );
        SparseColMat::<usize, f64>::new(
            // col_ptrs and row_indices were just produced into preallocated
            // buffers by permute_self_adjoint_to_unsorted from a valid n×n
            // symbolic CSC and a valid permutation. That routine writes an
            // unsorted CSC with col_ptrs length n + 1, monotone column ranges
            // within row_indices, and every row index in 0..n.
            // SAFETY: those are the hard SymbolicSparseColMat invariants; the
            // following faer symbolic Cholesky routines accept this unsorted
            // self-adjoint permutation.
            unsafe { SymbolicSparseColMat::new_unchecked(n, n, col_ptrs, None, row_indices) },
            values,
        )
    };

    // 3. Symbolic analysis
    let symbolic = {
        let mut mem = MemBuffer::new(StackReq::any_of(&[
            simplicial::prefactorize_symbolic_cholesky_scratch::<usize>(n, a_nnz),
            simplicial::factorize_simplicial_symbolic_cholesky_scratch::<usize>(n),
        ]));
        let stack = MemStack::new(&mut mem);
        let mut etree = vec![0isize; n];
        let mut col_counts = vec![0usize; n];
        let etree_ref = simplicial::prefactorize_symbolic_cholesky(
            &mut etree,
            &mut col_counts,
            a_perm_upper.symbolic(),
            stack,
        );
        simplicial::factorize_simplicial_symbolic_cholesky(
            a_perm_upper.symbolic(),
            etree_ref,
            &col_counts,
            stack,
        )
        .map_err(|_| LinalgError::ModelIsIllConditioned {
            condition_number: f64::INFINITY,
        })?
    };

    // 4. Numeric LLᵀ factorization
    let mut l_values = vec![0.0f64; symbolic.len_val()];
    {
        let mut mem = MemBuffer::new(simplicial::factorize_simplicial_numeric_llt_scratch::<
            usize,
            f64,
        >(n));
        simplicial::factorize_simplicial_numeric_llt::<usize, f64>(
            &mut l_values,
            a_perm_upper.as_ref(),
            LltRegularization::default(),
            &symbolic,
            MemStack::new(&mut mem),
        )
        .map_err(|_| LinalgError::HessianNotPositiveDefinite {
            min_eigenvalue: f64::NAN,
        })?;
    }

    // 5. Extract col_ptr, row_idx from the symbolic structure
    let l_col_ptr: Vec<usize> = symbolic.col_ptr().to_vec();
    let l_row_idx: Vec<usize> = symbolic.row_idx().to_vec();

    // 6. Compute logdet from L diagonal: L[j,j] = l_values[l_col_ptr[j]]
    let mut logdet = 0.0f64;
    for j in 0..n {
        let diag = l_values[l_col_ptr[j]];
        if diag <= 0.0 {
            return Err(LinalgError::HessianNotPositiveDefinite {
                min_eigenvalue: f64::NAN,
            });
        }
        logdet += diag.ln();
    }
    logdet *= 2.0;

    Ok(SimplicialFactor {
        l_col_ptr,
        l_row_idx,
        l_values,
        perm_inv,
        n,
        logdet,
    })
}

impl SimplicialFactor {
    /// Reconstruct the original-order dense SPD matrix represented by this
    /// permuted sparse Cholesky factor.
    ///
    /// The simplicial factor stores `L` for `P H Pᵀ = L Lᵀ`, with
    /// `perm_inv[original] = permuted`. We first assemble the dense permuted
    /// product and then map rows/columns back to the caller's coordinate order.
    fn assemble_h_dense_original_order(&self) -> Result<Array2<f64>, LinalgError> {
        if self.perm_inv.len() != self.n {
            bail_invalid_linalg!(
                "simplicial factor permutation length {} does not match dimension {}",
                self.perm_inv.len(),
                self.n
            );
        }
        let mut h_permuted = Array2::<f64>::zeros((self.n, self.n));
        for col in 0..self.n {
            let start = self.l_col_ptr[col];
            let end = self.l_col_ptr[col + 1];
            for left_idx in start..end {
                let left_row = self.l_row_idx[left_idx];
                let left_value = self.l_values[left_idx];
                if !left_value.is_finite() {
                    bail_invalid_linalg!(
                        "simplicial factor has non-finite L entry at value index {left_idx}"
                    );
                }
                for right_idx in start..end {
                    let right_row = self.l_row_idx[right_idx];
                    let right_value = self.l_values[right_idx];
                    h_permuted[[left_row, right_row]] += left_value * right_value;
                }
            }
        }

        let mut h_original = Array2::<f64>::zeros((self.n, self.n));
        for i in 0..self.n {
            let pi = self.perm_inv[i];
            if pi >= self.n {
                bail_invalid_linalg!(
                    "simplicial factor permutation maps row {i} to out-of-bounds index {pi}"
                );
            }
            for j in 0..self.n {
                let pj = self.perm_inv[j];
                if pj >= self.n {
                    bail_invalid_linalg!(
                        "simplicial factor permutation maps column {j} to out-of-bounds index {pj}"
                    );
                }
                let value = h_permuted[[pi, pj]];
                if !value.is_finite() {
                    bail_invalid_linalg!(
                        "dense reconstruction from sparse Cholesky produced non-finite values"
                    );
                }
                h_original[[i, j]] = value;
            }
        }
        Ok(h_original)
    }
}

/// Result of the Takahashi selected inversion.
///
/// Z stores entries of H⁻¹ at positions corresponding to the filled sparsity
/// pattern of the Cholesky factor L. Off-pattern entries are recovered exactly
/// on demand by cached column solves against the same simplicial factor.
pub struct TakahashiInverse {
    /// Z values stored in the same CSC pattern as L (lower triangular)
    z_values: Vec<f64>,
    /// Column pointers (owned copy from L)
    col_ptr: Vec<usize>,
    /// Row indices (owned copy from L)
    row_idx: Vec<usize>,
    /// Numeric values of the simplicial Cholesky factor L.
    l_values: Vec<f64>,
    /// Row-oriented access to L for forward solves in the permuted basis.
    rows_lower: Arc<Vec<Vec<(usize, f64)>>>,
    /// Exact inverse columns solved on demand for entries outside the selected
    /// inverse pattern. Keys are permuted-basis column indices.
    exact_columns: Mutex<BTreeMap<usize, Arc<Vec<f64>>>>,
    /// Inverse permutation returned by faer.
    perm_inv: Vec<usize>,
    /// Dimension
    n: usize,
}

impl TakahashiInverse {
    /// Binary search for entry (row, col) in lower-triangular CSC.
    /// Returns the value-array index if the entry exists.
    fn find_entry(col_ptr: &[usize], row_idx: &[usize], row: usize, col: usize) -> Option<usize> {
        let start = col_ptr[col];
        let end = col_ptr[col + 1];
        let slice = &row_idx[start..end];
        slice.binary_search(&row).ok().map(|pos| start + pos)
    }

    fn solve_permuted_column_from_cholesky(
        n: usize,
        col_ptr: &[usize],
        row_idx: &[usize],
        l_values: &[f64],
        rows_lower: &[Vec<(usize, f64)>],
        rhs_col: usize,
    ) -> Vec<f64> {
        let mut rhs = vec![0.0f64; n];
        rhs[rhs_col] = 1.0;
        let mut forward = vec![0.0f64; n];
        let mut solution = vec![0.0f64; n];

        for row in 0..n {
            let mut sum = rhs[row];
            let mut diag = None;
            for &(col, value) in &rows_lower[row] {
                if col < row {
                    sum -= value * forward[col];
                } else if col == row {
                    diag = Some(value);
                }
            }
            let l_rr = diag.expect("simplicial factor row should contain its diagonal");
            forward[row] = sum / l_rr;
        }

        for row in (0..n).rev() {
            let col_start = col_ptr[row];
            let col_end = col_ptr[row + 1];
            let mut sum = forward[row];
            let l_rr = l_values[col_start];
            for idx in (col_start + 1)..col_end {
                let lower_row = row_idx[idx];
                sum -= l_values[idx] * solution[lower_row];
            }
            solution[row] = sum / l_rr;
        }

        solution
    }

    fn exact_permuted_column(&self, col: usize) -> Arc<Vec<f64>> {
        {
            let cache = self
                .exact_columns
                .lock()
                .expect("exact Takahashi column cache mutex poisoned");
            if let Some(solution) = cache.get(&col) {
                return solution.clone();
            }
        }

        let solution = Arc::new(Self::solve_permuted_column_from_cholesky(
            self.n,
            &self.col_ptr,
            &self.row_idx,
            &self.l_values,
            self.rows_lower.as_ref(),
            col,
        ));

        let mut cache = self
            .exact_columns
            .lock()
            .expect("exact Takahashi column cache mutex poisoned");
        cache.entry(col).or_insert_with(|| solution.clone()).clone()
    }

    fn selected_value(
        z_values: &[f64],
        col_ptr: &[usize],
        row_idx: &[usize],
        row: usize,
        col: usize,
    ) -> Result<f64, LinalgError> {
        let (lower_row, lower_col) = if row >= col { (row, col) } else { (col, row) };
        Self::find_entry(col_ptr, row_idx, lower_row, lower_col)
            .map(|idx| z_values[idx])
            .ok_or_else(|| {
                LinalgError::InvalidInput(format!(
                    "simplicial selected-inverse pattern is missing entry ({lower_row},{lower_col})"
                ))
            })
    }

    /// Compute the selected inverse from a simplicial Cholesky factor.
    ///
    /// Given H = LLᵀ in the permuted basis, this applies the Takahashi
    /// recurrence on the filled Cholesky pattern. Off-pattern exact entries are
    /// recovered later by cached column solves from the same simplicial factor.
    pub fn compute(factor: &SimplicialFactor) -> Result<Self, LinalgError> {
        let n = factor.n;
        let col_ptr = factor.l_col_ptr.clone();
        let row_idx = factor.l_row_idx.clone();
        let nnz = factor.l_values.len();
        let mut z_values = vec![0.0f64; nnz];

        // Build row access for forward solves in the permuted basis.
        let mut rows_lower: Vec<Vec<(usize, f64)>> = vec![Vec::new(); n];
        for col in 0..n {
            for idx in col_ptr[col]..col_ptr[col + 1] {
                let row = row_idx[idx];
                rows_lower[row].push((col, factor.l_values[idx]));
            }
        }

        for j in (0..n).rev() {
            let diag_idx = col_ptr[j];
            let col_end = col_ptr[j + 1];
            let diag = factor.l_values[diag_idx];
            if !(diag.is_finite() && diag > 0.0) {
                return Err(LinalgError::HessianNotPositiveDefinite {
                    min_eigenvalue: f64::NAN,
                });
            }
            for idx in (diag_idx + 1)..col_end {
                let i = row_idx[idx];
                let mut correction = 0.0;
                for off_idx in (diag_idx + 1)..col_end {
                    let k = row_idx[off_idx];
                    let l_kj = factor.l_values[off_idx];
                    let z_ik = Self::selected_value(&z_values, &col_ptr, &row_idx, i, k)?;
                    correction += l_kj * z_ik;
                }
                let value = -correction / diag;
                if !value.is_finite() {
                    bail_invalid_linalg!(
                        "Takahashi selected inverse produced non-finite entry ({i},{j})"
                    );
                }
                z_values[idx] = value;
            }
            let mut correction = 0.0;
            for off_idx in (diag_idx + 1)..col_end {
                correction += factor.l_values[off_idx] * z_values[off_idx];
            }
            let value = (1.0 / diag - correction) / diag;
            if !value.is_finite() {
                bail_invalid_linalg!(
                    "Takahashi selected inverse produced non-finite diagonal entry ({j},{j})"
                );
            }
            z_values[diag_idx] = value;
        }

        Ok(TakahashiInverse {
            z_values,
            col_ptr,
            row_idx,
            l_values: factor.l_values.clone(),
            rows_lower: Arc::new(rows_lower),
            exact_columns: Mutex::new(BTreeMap::new()),
            perm_inv: factor.perm_inv.clone(),
            n,
        })
    }

    /// Get H⁻¹[i,j] in ORIGINAL (unpermuted) coordinates.
    pub fn get(&self, i: usize, j: usize) -> f64 {
        let pi = self.perm_inv[i];
        let pj = self.perm_inv[j];
        self.get_permuted(pi, pj)
    }

    /// Get Z[pi,pj] in permuted coordinates.
    fn get_permuted(&self, pi: usize, pj: usize) -> f64 {
        // Z is symmetric and stored as lower-triangular CSC.
        // Ensure row >= col for lookup.
        let (row, col) = if pi >= pj { (pi, pj) } else { (pj, pi) };
        if let Some(pos) = Self::find_entry(&self.col_ptr, &self.row_idx, row, col) {
            self.z_values[pos]
        } else {
            self.exact_permuted_column(col)[row]
        }
    }

    /// Diagonal of H⁻¹ in original ordering.
    pub fn diagonal(&self) -> Array1<f64> {
        Array1::from_iter((0..self.n).map(|i| self.get(i, i)))
    }

    /// H⁻¹[start..end, start..end] block in original ordering.
    pub fn block(&self, start: usize, end: usize) -> Array2<f64> {
        let dim = end - start;
        let mut out = Array2::zeros((dim, dim));
        for j_local in 0..dim {
            let j = start + j_local;
            for i_local in 0..dim {
                let i = start + i_local;
                out[[i_local, j_local]] = self.get(i, j);
            }
        }
        out
    }

    /// tr(H⁻¹ S) where S is given as sparse CSC, symmetric in either upper-
    /// triangle-only or full (both triangles stored) format.
    ///
    /// The algorithm iterates over the upper triangle of S (entries with
    /// row ≤ col), doubles off-diagonals, and skips lower-triangle entries.
    /// This is correct for both storage conventions:
    ///
    /// - **Upper-triangle-only** (for example, solver-owned sparse penalty blocks):
    ///   every off-diagonal pair has exactly one stored entry with row < col,
    ///   which we double.
    ///
    /// - **Full symmetric** (from `dense_to_sparse`): each off-diagonal pair
    ///   has entries at both (i,j) and (j,i).  We process only the row < col
    ///   entry and double it; the row > col mirror is skipped.  The diagonal
    ///   is stored once and counted once.
    ///
    /// In both cases: tr(Z S) = Σ_diag Z[i,i] S[i,i] + 2 Σ_{i<j} Z[i,j] S[i,j].
    pub fn trace_product_sparse(&self, s: &SparseColMat<usize, f64>) -> f64 {
        let (symbolic, values) = s.parts();
        let s_col_ptr = symbolic.col_ptr();
        let s_row_idx = symbolic.row_idx();
        let mut trace = 0.0;
        for col in 0..s.ncols() {
            let col_start = s_col_ptr[col];
            let col_end = s_col_ptr[col + 1];
            for idx in col_start..col_end {
                let row = s_row_idx[idx];
                if row > col {
                    continue; // skip lower triangle (handled via its mirror)
                }
                let val = values[idx];
                let z_ij = self.get(row, col);
                if row == col {
                    trace += z_ij * val;
                } else {
                    trace += 2.0 * z_ij * val;
                }
            }
        }
        trace
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::faer_ndarray::FaerCholesky;
    use ndarray::{array, Array1, Array2};

    fn approx_eq(a: f64, b: f64, tol: f64) {
        assert!(
            (a - b).abs() <= tol,
            "values differ: left={a:.12e}, right={b:.12e}, |diff|={:.12e}, tol={tol:.12e}",
            (a - b).abs()
        );
    }

    // ── dense_to_sparse ───────────────────────────────────────────────────

    #[test]
    fn dense_to_sparse_preserves_all_nonzero_entries() {
        // 3x3 matrix with a zero at (1,0) and all others nonzero.
        let m = array![[1.0, 2.0, 3.0], [0.0, 5.0, 6.0], [7.0, 8.0, 9.0]];
        let s = dense_to_sparse(&m, ZERO_TOL).unwrap();
        assert_eq!(s.nrows(), 3);
        assert_eq!(s.ncols(), 3);
        // 8 entries should be stored (one zero excluded).
        assert_eq!(s.compute_nnz(), 8);
    }

    #[test]
    fn dense_to_sparse_round_trips_via_matvec_identity() {
        // Verify that (sparse A) * e_j == column j of A for each column.
        let m = array![[4.0, 1.0, 0.5], [1.0, 3.0, 2.0], [0.5, 2.0, 6.0]];
        let s = dense_to_sparse(&m, ZERO_TOL).unwrap();
        for j in 0..3 {
            let mut ej = Array1::<f64>::zeros(3);
            ej[j] = 1.0;
            // Multiply via the raw faer sparse multiply.
            let result = {
                let mut out = Array1::<f64>::zeros(3);
                let (sym, vals) = s.parts();
                let col_ptr = sym.col_ptr();
                let row_idx = sym.row_idx();
                for col in 0..3 {
                    for idx in col_ptr[col]..col_ptr[col + 1] {
                        let row = row_idx[idx];
                        out[row] += vals[idx] * ej[col];
                    }
                }
                out
            };
            for i in 0..3 {
                approx_eq(result[i], m[[i, j]], 1e-14);
            }
        }
    }

    #[test]
    fn dense_to_sparse_filters_entries_below_tolerance() {
        let tol = 0.1;
        let m = array![[1.0, 0.05], [0.05, 2.0]];
        let s = dense_to_sparse(&m, tol).unwrap();
        // Only the two diagonal entries exceed tol.
        assert_eq!(s.compute_nnz(), 2, "off-diagonal entries below tol must be dropped");
    }

    // ── dense_to_sparse_symmetric_upper ───────────────────────────────────

    #[test]
    fn dense_to_sparse_symmetric_upper_stores_only_upper_triangle() {
        // Full symmetric 3x3 matrix — only upper triangle (i<=j) should be stored.
        let m = array![[4.0, 1.0, 2.0], [1.0, 5.0, 3.0], [2.0, 3.0, 6.0]];
        let s = dense_to_sparse_symmetric_upper(&m, ZERO_TOL).unwrap();
        // Upper triangle has 3 diagonal + 3 off-diagonal = 6 entries.
        assert_eq!(s.compute_nnz(), 6);
    }

    // ── sparse_symmetric_upper_matvec_public ──────────────────────────────

    #[test]
    fn sparse_symmetric_upper_matvec_matches_dense_matvec() {
        // Symmetric matrix A; upper-sparse encodes only the upper triangle.
        // A * v must equal the result of the symmetric matvec.
        let a = array![[4.0, 2.0, 0.0], [2.0, 5.0, 3.0], [0.0, 3.0, 6.0]];
        let v = array![1.0, 2.0, 3.0];
        let expected = a.dot(&v); // dense reference
        let a_sparse = dense_to_sparse_symmetric_upper(&a, ZERO_TOL).unwrap();
        let got = sparse_symmetric_upper_matvec_public(&a_sparse, &v);
        for i in 0..3 {
            approx_eq(got[i], expected[i], 1e-13);
        }
    }

    #[test]
    fn sparse_symmetric_upper_matvec_diagonal_only() {
        // Pure diagonal matrix: matvec should scale each component.
        let a = array![[3.0, 0.0, 0.0], [0.0, 5.0, 0.0], [0.0, 0.0, 7.0]];
        let v = array![2.0, 4.0, 6.0];
        let a_sparse = dense_to_sparse_symmetric_upper(&a, ZERO_TOL).unwrap();
        let got = sparse_symmetric_upper_matvec_public(&a_sparse, &v);
        approx_eq(got[0], 6.0, 1e-14);
        approx_eq(got[1], 20.0, 1e-14);
        approx_eq(got[2], 42.0, 1e-14);
    }

    // ── solve_sparse_spd / logdet_from_factor ─────────────────────────────

    #[test]
    fn solve_sparse_spd_recovers_known_solution() {
        // A = [[4,2],[2,5]]; A^{-1} b = [0.5, 2.0] for b = [6, 11].
        let a = array![[4.0, 2.0], [2.0, 5.0]];
        let a_sparse = dense_to_sparse_symmetric_upper(&a, ZERO_TOL).unwrap();
        let factor = factorize_sparse_spd(&a_sparse).unwrap();
        let rhs = array![6.0, 11.0];
        let x = solve_sparse_spd(&factor, &rhs).unwrap();
        // A^-1 = (1/16)*[[5,-2],[-2,4]]; x = (1/16)*[5*6-2*11, -2*6+4*11] = [0.5, 2.0]
        approx_eq(x[0], 0.5, 1e-12);
        approx_eq(x[1], 2.0, 1e-12);
    }

    #[test]
    fn solve_sparse_spd_3x3_round_trip() {
        let a: Array2<f64> = array![
            [9.0, 3.0, 1.0],
            [3.0, 8.0, 2.0],
            [1.0, 2.0, 7.0]
        ];
        let a_sparse = dense_to_sparse_symmetric_upper(&a, ZERO_TOL).unwrap();
        let factor = factorize_sparse_spd(&a_sparse).unwrap();
        for j in 0..3 {
            let mut ej = Array1::<f64>::zeros(3);
            ej[j] = 1.0;
            let col_j = solve_sparse_spd(&factor, &ej).unwrap();
            // A * x should equal ej.
            let ax = a.dot(&col_j);
            for i in 0..3 {
                approx_eq(ax[i], ej[i], 1e-12);
            }
        }
    }

    #[test]
    fn logdet_from_factor_matches_dense_logdet_diagonal() {
        // Diagonal matrix diag(4,9,16): log-det = log(4)+log(9)+log(16)
        let a: Array2<f64> =
            array![[4.0, 0.0, 0.0], [0.0, 9.0, 0.0], [0.0, 0.0, 16.0]];
        let a_sparse = dense_to_sparse_symmetric_upper(&a, ZERO_TOL).unwrap();
        let factor = factorize_sparse_spd(&a_sparse).unwrap();
        let logdet = logdet_from_factor(&factor).unwrap();
        let expected = 4.0_f64.ln() + 9.0_f64.ln() + 16.0_f64.ln();
        approx_eq(logdet, expected, 1e-12);
    }

    #[test]
    fn logdet_from_factor_matches_2x2_formula() {
        // A = [[4,2],[2,5]]; det(A) = 20-4 = 16; log-det = log(16)
        let a = array![[4.0, 2.0], [2.0, 5.0]];
        let a_sparse = dense_to_sparse_symmetric_upper(&a, ZERO_TOL).unwrap();
        let factor = factorize_sparse_spd(&a_sparse).unwrap();
        let logdet = logdet_from_factor(&factor).unwrap();
        approx_eq(logdet, 16.0_f64.ln(), 1e-12);
    }

    #[test]
    fn solve_sparse_spd_dimension_mismatch_returns_error() {
        let a = array![[4.0, 2.0], [2.0, 5.0]];
        let a_sparse = dense_to_sparse_symmetric_upper(&a, ZERO_TOL).unwrap();
        let factor = factorize_sparse_spd(&a_sparse).unwrap();
        let rhs = array![1.0, 2.0, 3.0]; // wrong length
        assert!(solve_sparse_spd(&factor, &rhs).is_err());
    }

    #[test]
    fn takahashi_diagonal_matches_dense_inverse() {
        // 4x4 SPD matrix
        let h = array![
            [4.0, 0.2, 0.0, 0.0],
            [0.2, 3.0, 0.1, 0.0],
            [0.0, 0.1, 2.5, 0.3],
            [0.0, 0.0, 0.3, 2.0]
        ];
        let h_sparse = dense_to_sparse_symmetric_upper(&h, ZERO_TOL).unwrap();

        // Dense inverse for reference via column solves
        let chol = h.cholesky(Side::Lower).unwrap();
        let mut h_inv = Array2::<f64>::zeros((4, 4));
        for j in 0..4 {
            let mut rhs = Array1::<f64>::zeros(4);
            rhs[j] = 1.0;
            let col = chol.solvevec(&rhs);
            for i in 0..4 {
                h_inv[[i, j]] = col[i];
            }
        }

        let sfactor = factorize_simplicial(&h_sparse).unwrap();
        let taka = TakahashiInverse::compute(&sfactor).unwrap();
        let diag = taka.diagonal();

        // Diagonal of selected inverse should match dense inverse diagonal
        for i in 0..4 {
            approx_eq(diag[i], h_inv[[i, i]], 1e-10);
        }
    }

    #[test]
    fn takahashi_logdet_matches_dense() {
        let h = array![
            [4.0, 0.2, 0.0, 0.0],
            [0.2, 3.0, 0.1, 0.0],
            [0.0, 0.1, 2.5, 0.3],
            [0.0, 0.0, 0.3, 2.0]
        ];
        let h_sparse = dense_to_sparse_symmetric_upper(&h, ZERO_TOL).unwrap();

        // Dense logdet via existing factor
        let existing = factorize_sparse_spd(&h_sparse).unwrap();
        let logdet_dense = existing.logdet;

        let sfactor = factorize_simplicial(&h_sparse).unwrap();
        approx_eq(sfactor.logdet, logdet_dense, 1e-10);
    }

    #[test]
    fn takahashi_get_and_block_recover_off_pattern_inverse_entries() {
        let h = array![
            [4.0, 1.0, 0.0, 0.0],
            [1.0, 3.0, 1.0, 0.0],
            [0.0, 1.0, 2.5, 1.0],
            [0.0, 0.0, 1.0, 2.0]
        ];
        let h_sparse = dense_to_sparse_symmetric_upper(&h, ZERO_TOL).unwrap();

        let chol = h.cholesky(Side::Lower).unwrap();
        let mut h_inv = Array2::<f64>::zeros((4, 4));
        for j in 0..4 {
            let mut rhs = Array1::<f64>::zeros(4);
            rhs[j] = 1.0;
            let col = chol.solvevec(&rhs);
            for i in 0..4 {
                h_inv[[i, j]] = col[i];
            }
        }

        let sfactor = factorize_simplicial(&h_sparse).unwrap();
        let taka = TakahashiInverse::compute(&sfactor).unwrap();

        assert!(
            h_inv[[0, 2]].abs() > 1e-8,
            "reference off-pattern inverse entry should be nonzero"
        );
        approx_eq(taka.get(0, 2), h_inv[[0, 2]], 1e-10);

        let block = taka.block(0, 3);
        approx_eq(block[[0, 2]], h_inv[[0, 2]], 1e-10);
        approx_eq(block[[2, 0]], h_inv[[2, 0]], 1e-10);
    }
}
