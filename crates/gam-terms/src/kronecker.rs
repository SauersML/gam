//! λ-invariant Kronecker tensor structure.
//!
//! Every numeric primitive this module needs — the faer/ndarray bridges, the
//! symmetric sanitiser, the robust marginal eigensolve and the multi-index
//! walk — is owned by [`crate::construction`], which also owns the per-iterate
//! Kronecker engine that consumes this structure. This file used to carry a
//! second copy of all of them, differing only in the error type it wrapped
//! them in (#2470); the eigensolve failure is an estimation failure and is now
//! reported as one on both routes.

use crate::construction::{kronecker_marginal_eigensystems, kronecker_multi_index_advance};
use gam_problem::EstimationError;
use ndarray::{Array1, Array2};
use std::sync::Arc;

pub use gam_problem::penalty_matrix::kronecker_product;

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
    ) -> Result<Self, EstimationError> {
        let d = marginal_dims.len();
        // Eigendecompose each marginal penalty once through the same robust path
        // used by KroneckerPenaltySystem so every Kronecker caller sees the same
        // eigensystem and pseudo-logdet surface.
        let mut marginal_eigenvalues = Vec::with_capacity(d);
        let mut marginal_qs = Vec::with_capacity(d);
        for (evals, evecs) in kronecker_marginal_eigensystems(
            marginal_penalties,
            "KroneckerInvariantStructure::compute",
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
        // A marginal with an all-zero penalty (‖S_k‖_F = 0) has no spectrum
        // to balance and contributes nothing to the balanced eigenvalue; it is
        // skipped rather than divided by a floor.
        let frob_norms: Vec<f64> = marginal_penalties
            .iter()
            .map(|s| s.iter().map(|v| v * v).sum::<f64>().sqrt())
            .collect();
        loop {
            let mut sigma = 0.0;
            for k in 0..d {
                if frob_norms[k] > 0.0 {
                    sigma += marginal_eigenvalues[k][multi_idx[k]] / frob_norms[k];
                }
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
