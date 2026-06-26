use crate::basis::BasisError;
use faer::{Mat, MatRef, Side};
use gam_linalg::faer_ndarray::FaerLinalgError;
use ndarray::{Array1, Array2, Axis};
use rayon::prelude::*;
use std::sync::Arc;

fn array_to_faer(array: &Array2<f64>) -> Mat<f64> {
    let (rows, cols) = array.dim();
    Mat::from_fn(rows, cols, |i, j| array[[i, j]])
}

fn mat_to_array(mat: &Mat<f64>) -> Array2<f64> {
    let mut out = Array2::<f64>::zeros((mat.nrows(), mat.ncols()));
    for i in 0..mat.nrows() {
        for j in 0..mat.ncols() {
            out[[i, j]] = mat[(i, j)];
        }
    }
    out
}

fn mat_max_abs_element(matrix: MatRef<'_, f64>) -> f64 {
    let (rows, cols) = matrix.shape();
    let mut maxval = 0.0_f64;
    for i in 0..rows {
        for j in 0..cols {
            let val = matrix[(i, j)];
            if val.is_finite() {
                maxval = maxval.max(val.abs());
            }
        }
    }
    maxval
}

fn sanitize_symmetric_faer(matrix: &Mat<f64>) -> Mat<f64> {
    let (rows, cols) = matrix.as_ref().shape();
    assert_eq!(rows, cols, "Matrix must be square for sanitization");

    let mut sanitized = matrix.clone();

    for i in 0..rows {
        let diag = sanitized[(i, i)];
        if !diag.is_finite() {
            sanitized[(i, i)] = 0.0;
        }
        for j in (i + 1)..cols {
            let mut upper = sanitized[(i, j)];
            let mut lower = sanitized[(j, i)];
            if !upper.is_finite() {
                upper = 0.0;
            }
            if !lower.is_finite() {
                lower = 0.0;
            }
            let avg = 0.5 * (upper + lower);
            sanitized[(i, j)] = avg;
            sanitized[(j, i)] = avg;
        }
    }

    let scale = mat_max_abs_element(sanitized.as_ref());
    let tiny = (scale * 1e-14).max(1e-30);
    for i in 0..rows {
        for j in 0..cols {
            let val = sanitized[(i, j)];
            if !val.is_finite() {
                sanitized[(i, j)] = 0.0;
            } else if val.abs() < tiny {
                sanitized[(i, j)] = 0.0;
            }
        }
    }

    sanitized
}

/// Strict spectral classifier used as a final guard on penalty eigendecompositions.
///
/// Penalty matrices fed to the GAM solver are required to be PSD by construction.
/// This routine snaps roundoff-zero eigenvalues to exact zero, accepts strictly
/// positive eigenvalues, and rejects materially-indefinite or non-finite spectra
/// with a hard error rather than silently rewriting them. The previous behaviour
/// (mass-zeroing negative or non-finite eigenvalues) hid construction bugs and
/// changed the optimisation objective downstream.
///
/// `C_EPS_P_FACTOR = 64` chooses the multiplier `c` in
/// `tol = c * eps_machine * p * scale`: 64 absorbs the rounding accumulated in a
/// symmetric eigendecomposition of a moderate-dimension matrix while still
/// rejecting the 1e-12 * scale magnitudes that previously slipped through.
fn classify_eigenvalues_strict(eigenvalues: &mut [f64], context: &str) -> Result<(), BasisError> {
    const C_EPS_P_FACTOR: f64 = 64.0;
    let p = eigenvalues.len();

    let mut scale = 0.0_f64;
    for (idx, &val) in eigenvalues.iter().enumerate() {
        if !val.is_finite() {
            return Err(BasisError::Other(format!(
                "Penalty spectrum check failed in '{context}': non-finite eigenvalue {value:?} at index {index}",
                value = val,
                index = idx
            )));
        }
        scale = scale.max(val.abs());
    }

    // p * eps captures the rounding floor of a symmetric eigendecomposition of a
    // p-dimensional matrix; multiplying by `scale` lifts the floor to the actual
    // magnitude of the spectrum. The constant `C_EPS_P_FACTOR` provides headroom
    // for the residual rounding in subsequent matmuls.
    let tolerance =
        (C_EPS_P_FACTOR * f64::EPSILON * (p.max(1) as f64) * scale).max(f64::MIN_POSITIVE);

    for (idx, val) in eigenvalues.iter_mut().enumerate() {
        if val.abs() <= tolerance {
            *val = 0.0;
        } else if *val < 0.0 {
            return Err(BasisError::Other(format!(
                "Penalty spectrum check failed in '{context}': indefinite eigenvalue {value:.3e} at index {index} (tolerance {tolerance:.3e}, scale {scale:.3e})",
                value = *val,
                index = idx
            )));
        }
    }
    Ok(())
}

fn robust_eighwith_policy<M, V, E, Validate, Sanitize, EigCall, MapErr>(
    matrix: &M,
    context: &str,
    validate_input: Validate,
    sanitize: Sanitize,
    mut eig_call: EigCall,
    map_error: MapErr,
) -> Result<(Vec<f64>, V), BasisError>
where
    Validate: Fn(&M, &str) -> Result<(), BasisError>,
    Sanitize: Fn(&M) -> M,
    EigCall: FnMut(&M) -> Result<(Vec<f64>, V), E>,
    MapErr: Fn(E, &str) -> BasisError,
{
    validate_input(matrix, context)?;

    // The sanitize step only enforces exact symmetry by averaging M and M^T and
    // zeros sub-eps noise; it never adds a diagonal ridge. Adding ridge changes
    // the matrix being decomposed, which silently changes the optimisation
    // objective downstream. If eigh genuinely fails on a finite symmetric input,
    // surface the error instead of mutating the spectrum.
    let candidate = sanitize(matrix);
    match eig_call(&candidate) {
        Ok((mut eigenvalues, eigenvectors)) => {
            classify_eigenvalues_strict(&mut eigenvalues, context)?;
            Ok((eigenvalues, eigenvectors))
        }
        Err(err) => Err(map_error(err, context)),
    }
}

fn robust_eigh_faer(
    matrix: &Mat<f64>,
    side: Side,
    context: &str,
) -> Result<(Vec<f64>, Mat<f64>), BasisError> {
    robust_eighwith_policy(
        matrix,
        context,
        |mat, ctx| {
            let (rows, cols) = mat.as_ref().shape();
            for i in 0..rows {
                for j in 0..cols {
                    let val = mat[(i, j)];
                    if !val.is_finite() {
                        let max_abs = mat_max_abs_element(mat.as_ref());
                        return Err(BasisError::Other(format!(
                            "{} contains non-finite entries (max finite magnitude {:.3e})",
                            ctx, max_abs
                        )));
                    }
                }
            }
            Ok(())
        },
        sanitize_symmetric_faer,
        |candidate| {
            let eig = candidate.as_ref().self_adjoint_eigen(side)?;
            let diag = eig.S();
            let mut eigenvalues = Vec::with_capacity(diag.dim());
            for idx in 0..diag.dim() {
                eigenvalues.push(diag[idx]);
            }

            let vectors_ref = eig.U();
            let mut eigenvectors = Mat::<f64>::zeros(vectors_ref.nrows(), vectors_ref.ncols());
            for i in 0..vectors_ref.nrows() {
                for j in 0..vectors_ref.ncols() {
                    eigenvectors[(i, j)] = vectors_ref[(i, j)];
                }
            }
            Ok((eigenvalues, eigenvectors))
        },
        |err, _ctx| {
            BasisError::Other(format!(
                "Eigendecomposition failed: {}",
                FaerLinalgError::SelfAdjointEigen(err)
            ))
        },
    )
}

fn robust_eigh(
    matrix: &Array2<f64>,
    side: Side,
    context: &str,
) -> Result<(Array1<f64>, Array2<f64>), BasisError> {
    let matrix_faer = array_to_faer(matrix);
    let (eigenvalues, eigenvectors) = robust_eigh_faer(&matrix_faer, side, context)?;
    Ok((Array1::from_vec(eigenvalues), mat_to_array(&eigenvectors)))
}

fn kronecker_marginal_eigensystems(
    marginal_penalties: &[Array2<f64>],
    context: &str,
) -> Result<Vec<(Array1<f64>, Array2<f64>)>, BasisError> {
    let mut eigensystems = Vec::with_capacity(marginal_penalties.len());
    for (k, penalty) in marginal_penalties.iter().enumerate() {
        eigensystems.push(robust_eigh(
            penalty,
            Side::Lower,
            &format!("{context} marginal {k}"),
        )?);
    }
    Ok(eigensystems)
}

/// Computes the Kronecker product A ⊗ B for penalty matrix construction.
/// This is used to create tensor product penalties that enforce smoothness
/// in multiple dimensions for interaction terms.
pub fn kronecker_product(a: &Array2<f64>, b: &Array2<f64>) -> Array2<f64> {
    let (arows, a_cols) = a.dim();
    let (brows, b_cols) = b.dim();
    if arows == 0 || a_cols == 0 || brows == 0 || b_cols == 0 {
        return Array2::zeros((arows * brows, a_cols * b_cols));
    }
    let mut result = Array2::zeros((arows * brows, a_cols * b_cols));

    result
        .axis_chunks_iter_mut(Axis(0), brows)
        .into_par_iter()
        .enumerate()
        .for_each(|(i, mut row_block)| {
            let arow = a.row(i);
            let col_chunks = row_block.axis_chunks_iter_mut(Axis(1), b_cols);
            for (j, mut block) in col_chunks.into_iter().enumerate() {
                let aval = arow[j];
                if aval == 0.0 {
                    continue;
                }
                for (dest, &src) in block.iter_mut().zip(b.iter()) {
                    *dest = aval * src;
                }
            }
        });

    result
}

/// Advance a row-major multi-index over the `dims` grid in place.
/// Returns `true` when the grid is exhausted (the index wrapped back to all-zero).
#[inline]
fn kronecker_multi_index_advance(multi_idx: &mut [usize], dims: &[usize]) -> bool {
    let mut carry = true;
    for dim in (0..dims.len()).rev() {
        if carry {
            multi_idx[dim] += 1;
            if multi_idx[dim] < dims[dim] {
                carry = false;
            } else {
                multi_idx[dim] = 0;
            }
        }
    }
    carry
}

/// λ-invariant Kronecker tensor structure: everything in a tensor-product fit
/// that depends ONLY on the marginal designs/penalties (which are fixed for the
/// whole fit) and NOT on the smoothing parameters λ = exp(ρ).
///
/// The marginal eigendecomposition (`O(Σ q_k³)`), the reparameterized marginals
/// `B_k · U_k`, and the balanced-penalty shrinkage scale `max_bal` are all
/// functions of the fixed marginal data alone. Caching them once per fit lets
/// every outer REML iterate (50+ per fit on the #1082 tensor cases) skip the
/// repeated `eigh()` calls and `B_k U_k` GEMMs; only the cheap
/// `kronecker_logdet_and_derivatives` λ-grid sweep is redone per iterate.
#[derive(Clone, Debug)]
pub struct KroneckerInvariantStructure {
    /// Marginal eigenvalues from each marginal penalty eigendecomposition.
    ///
    /// `Arc`-shared so handing this structure to the per-iterate memoized
    /// engine is an O(1) refcount bump, not a deep array copy.
    pub marginal_eigenvalues: Arc<Vec<Array1<f64>>>,
    /// Marginal eigenvector matrices U_k.
    pub marginal_qs: Arc<Vec<Array2<f64>>>,
    /// Reparameterized marginal designs: `B_k · U_k` for each marginal k.
    pub reparameterized_marginals: Arc<Vec<Array2<f64>>>,
    /// Max balanced-penalty eigenvalue scale `max_k-grid Σ_k μ_{k,j_k}/||S_k||_F`,
    /// used to form the shrinkage ridge `floor * max_bal`. λ-independent.
    pub max_balanced_eigenvalue: f64,
}

impl KroneckerInvariantStructure {
    /// Compute the λ-invariant tensor structure once from the fixed marginal data.
    pub fn compute(
        marginal_designs: &[Array2<f64>],
        marginal_penalties: &[Array2<f64>],
        marginal_dims: &[usize],
    ) -> Result<Self, BasisError> {
        let d = marginal_dims.len();
        // Eigendecompose each marginal penalty once through the same robust path
        // used by KroneckerPenaltySystem so every Kronecker caller sees the same
        // eigensystem and pseudo-logdet surface.
        let mut marginal_eigenvalues = Vec::with_capacity(d);
        let mut marginal_qs = Vec::with_capacity(d);
        for (evals, evecs) in kronecker_marginal_eigensystems(
            marginal_penalties,
            "kronecker_reparameterization_engine",
        )? {
            marginal_eigenvalues.push(evals);
            marginal_qs.push(evecs);
        }

        // Reparameterized marginals: B_k · U_k.
        let reparameterized_marginals: Vec<Array2<f64>> = marginal_designs
            .iter()
            .zip(marginal_qs.iter())
            .map(|(b_k, u_k)| gam_linalg::faer_ndarray::fast_ab(b_k, u_k))
            .collect();

        // Max balanced eigenvalue: for Kronecker, the balanced penalty's max
        // eigenvalue is the max over multi-indices of Σ_k (1/||S_k||_F) μ_{k,j_k}.
        let mut max_balanced_eigenvalue = 0.0_f64;
        let mut multi_idx = vec![0usize; d];
        let frob_norms: Vec<f64> = marginal_penalties
            .iter()
            .map(|s| s.iter().map(|v| v * v).sum::<f64>().sqrt().max(1e-12))
            .collect();
        loop {
            let mut sigma = 0.0;
            for k in 0..d {
                sigma += marginal_eigenvalues[k][multi_idx[k]] / frob_norms[k];
            }
            max_balanced_eigenvalue = max_balanced_eigenvalue.max(sigma);

            if kronecker_multi_index_advance(&mut multi_idx, marginal_dims) {
                break;
            }
        }

        Ok(Self {
            marginal_eigenvalues: Arc::new(marginal_eigenvalues),
            marginal_qs: Arc::new(marginal_qs),
            reparameterized_marginals: Arc::new(reparameterized_marginals),
            max_balanced_eigenvalue,
        })
    }
}
