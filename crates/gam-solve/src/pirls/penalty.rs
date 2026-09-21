use faer::sparse::SparseRowMat;
use gam_linalg::faer_ndarray::{fast_ab, fast_atb, fast_atv, fast_av};
use gam_linalg::matrix::DesignMatrix;
use ndarray::{Array1, Array2};
use std::sync::Arc;

/// Coordinate frame for PIRLS inner iteration.
pub(crate) enum WorkingCoordinateDesign {
    OriginalSparseNative,
    TransformedExplicit {
        x_transformed: DesignMatrix,
        x_csr: Option<SparseRowMat<usize, f64>>,
    },
    TransformedImplicit {
        transform: WorkingReparamTransform,
    },
}

#[derive(Clone)]
pub(crate) enum WorkingReparamTransform {
    Dense(Arc<Array2<f64>>),
}

impl WorkingReparamTransform {
    pub(super) fn apply(&self, vector: &Array1<f64>) -> Array1<f64> {
        match self {
            Self::Dense(qs) => fast_av(qs.as_ref(), vector),
        }
    }

    pub(super) fn apply_transpose(&self, vector: &Array1<f64>) -> Array1<f64> {
        match self {
            Self::Dense(qs) => fast_atv(qs, vector),
        }
    }

    pub(super) fn materialize_dense(&self) -> Array2<f64> {
        match self {
            Self::Dense(qs) => qs.as_ref().clone(),
        }
    }

    pub(super) fn conjugate_matrix(&self, matrix: &Array2<f64>) -> Array2<f64> {
        match self {
            Self::Dense(qs) => {
                let tmp = fast_atb(qs, matrix);
                symmetrize_dense_matrix(&fast_ab(&tmp, qs))
            }
        }
    }
}

#[derive(Clone)]
pub(crate) enum PirlsPenalty {
    Dense {
        s_transformed: Array2<f64>,
        e_transformed: Array2<f64>,
    },
}

impl PirlsPenalty {
    pub(super) fn dim(&self) -> usize {
        match self {
            Self::Dense { s_transformed, .. } => s_transformed.ncols(),
        }
    }

    pub(super) fn rank(&self) -> usize {
        match self {
            Self::Dense { e_transformed, .. } => e_transformed.nrows(),
        }
    }

    /// Whether assembling `S = E' E` has squared the penalty-root condition
    /// number far enough to discard more than half of binary64's significant
    /// digits.  Above `1/sqrt(eps)` in row energy, a Cholesky solve of the Gram
    /// is numerically a different problem from a QR solve of its PSD root.
    ///
    /// Reparameterized dense penalties store mutually orthogonal spectral-root
    /// rows, so their squared row norms are exactly the represented positive
    /// eigenvalues.
    pub(super) fn requires_root_solve(&self, stabilizing_floor: f64) -> bool {
        let Self::Dense { e_transformed, .. } = self;
        let mut min_positive = if stabilizing_floor.is_finite() && stabilizing_floor > 0.0 {
            stabilizing_floor
        } else {
            f64::INFINITY
        };
        let mut max_energy = if stabilizing_floor.is_finite() && stabilizing_floor > 0.0 {
            stabilizing_floor
        } else {
            0.0
        };
        for row in e_transformed.rows() {
            let energy = row.dot(&row);
            if energy.is_infinite() {
                return true;
            }
            if energy > 0.0 && energy.is_finite() {
                min_positive = min_positive.min(energy);
                max_energy = max_energy.max(energy);
            }
        }
        min_positive.is_finite() && max_energy / min_positive > f64::EPSILON.sqrt().recip()
    }

    pub(super) fn write_root_rows(&self, out: &mut Array2<f64>, first_row: usize) {
        match self {
            Self::Dense { e_transformed, .. } => {
                let end = first_row + e_transformed.nrows();
                out.slice_mut(ndarray::s![first_row..end, ..])
                    .assign(e_transformed);
            }
        }
    }

    /// Write the penalty residual `q = E beta` whose normal-equation image is
    /// the exact penalty gradient: `E' q = S beta`.
    ///
    /// Keeping this residual in root space lets the stiff-penalty PIRLS path
    /// solve the augmented least-squares problem directly. Forming `S beta` in
    /// coefficient space first would lose precisely the stationarity digits
    /// that the root solve is meant to preserve.
    pub(super) fn write_root_residual(
        &self,
        beta: &Array1<f64>,
        out: &mut Array1<f64>,
        first_row: usize,
    ) {
        match self {
            Self::Dense { e_transformed, .. } => {
                let e_beta = fast_av(e_transformed, beta);
                out.slice_mut(ndarray::s![first_row..first_row + e_beta.len()])
                    .assign(&e_beta);
            }
        }
    }

    pub(super) fn add_to_hessian(&self, hessian: &mut Array2<f64>) {
        match self {
            Self::Dense { s_transformed, .. } => {
                *hessian += s_transformed;
            }
        }
    }

    pub(super) fn apply(&self, beta: &Array1<f64>) -> Array1<f64> {
        match self {
            Self::Dense { e_transformed, .. } => {
                // Apply the dense penalty through its square root rather than
                // through the assembled Gram matrix:
                //
                //     S beta = E' (E beta),  S = E' E.
                //
                // Forming `S` is unavoidable for the direct Hessian solve, but
                // using it again for the gradient squares the conditioning of
                // `E`.  At wide smoothing-parameter ratios a coefficient can
                // have large cancelling coordinates, so `beta.dot(S beta)` can
                // even become negative although the represented penalty is
                // positive semidefinite.  Keeping value and gradient on the
                // root representation makes them one coherent numerical atom.
                let e_beta = fast_av(e_transformed, beta);
                fast_atv(e_transformed, &e_beta)
            }
        }
    }

    /// The penalty energy `beta' S beta`, evaluated as `||E beta||²` on the
    /// root for the same reason as [`Self::apply`].
    pub(super) fn quadratic(&self, beta: &Array1<f64>) -> f64 {
        match self {
            Self::Dense { e_transformed, .. } => {
                let e_beta = fast_av(e_transformed, beta);
                e_beta.dot(&e_beta)
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::PirlsPenalty;
    use ndarray::{Array1, array};

    #[test]
    fn dense_penalty_value_and_gradient_use_the_psd_root() {
        // The small eigen-direction is exactly representable in E, but is lost
        // when E' E is rounded: every entry of the Gram rounds to 1e32.  This is
        // the cancellation pattern reached by stiff outer-REML trial points in
        // #2316.  The represented penalty is nevertheless unambiguously
        // ||E beta||^2 = 4 with gradient E'(E beta) = [2, -2].
        let e_transformed = array![[1.0e16, 1.0e16], [1.0, -1.0]];
        let s_transformed = e_transformed.t().dot(&e_transformed);
        let penalty = PirlsPenalty::Dense {
            s_transformed,
            e_transformed,
        };
        let beta = array![1.0, -1.0];

        assert_eq!(penalty.quadratic(&beta), 4.0);
        assert_eq!(penalty.apply(&beta), array![2.0, -2.0]);
    }

    #[test]
    fn root_solve_gate_is_derived_from_gram_precision_loss() {
        let stiff = PirlsPenalty::Dense {
            s_transformed: array![[1.0e10, 0.0], [0.0, 1.0]],
            e_transformed: array![[1.0e5, 0.0], [0.0, 1.0]],
        };
        let ordinary = PirlsPenalty::Dense {
            s_transformed: array![[1.0e6, 0.0], [0.0, 1.0]],
            e_transformed: array![[1.0e3, 0.0], [0.0, 1.0]],
        };

        assert!(stiff.requires_root_solve(0.0));
        assert!(!ordinary.requires_root_solve(0.0));

        let rank_one = PirlsPenalty::Dense {
            s_transformed: array![[1.0e10, 1.0e10], [1.0e10, 1.0e10]],
            e_transformed: array![[1.0e5, 1.0e5]],
        };
        assert!(rank_one.requires_root_solve(1.0));
        assert!(!rank_one.requires_root_solve(1.0e4));
    }

    #[test]
    fn root_residual_maps_to_penalty_gradient() {
        let root = array![[3.0, 0.0], [0.0, 2.0]];
        let penalty = PirlsPenalty::Dense {
            s_transformed: root.t().dot(&root),
            e_transformed: root.clone(),
        };
        let beta = array![0.25, -0.75];
        let mut residual = Array1::<f64>::zeros(4);

        penalty.write_root_residual(&beta, &mut residual, 2);
        let mapped = root.t().dot(&residual.slice(ndarray::s![2..]).to_owned());

        assert_eq!(mapped, penalty.apply(&beta));
    }
}

#[inline]
pub(super) fn symmetrize_dense_matrix(matrix: &Array2<f64>) -> Array2<f64> {
    (matrix + &matrix.t().to_owned()) * 0.5
}
