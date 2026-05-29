use std::fmt;

use ndarray::{Array1, Array2, ArrayView1, ArrayView2};

pub const GEOMETRY_EPS: f64 = 1.0e-12;

#[derive(Debug, Clone, PartialEq)]
pub enum GeometryError {
    DimensionMismatch {
        context: &'static str,
        expected: usize,
        got: usize,
    },
    InvalidPoint(&'static str),
    Singular(&'static str),
    /// A manifold primitive has no implementation for this manifold and must
    /// not silently fall back to a wrong default (e.g. a curved-manifold VJP
    /// for which no closed form is wired up yet).
    Unsupported(&'static str),
}

impl fmt::Display for GeometryError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::DimensionMismatch {
                context,
                expected,
                got,
            } => write!(f, "{context} expected length {expected}, got {got}"),
            Self::InvalidPoint(message) => write!(f, "invalid manifold point: {message}"),
            Self::Singular(message) => write!(f, "singular geometry operation: {message}"),
            Self::Unsupported(message) => write!(f, "unsupported geometry operation: {message}"),
        }
    }
}

impl std::error::Error for GeometryError {}

pub type GeometryResult<T> = Result<T, GeometryError>;

pub trait RiemannianManifold: Send + Sync {
    fn dim(&self) -> usize;

    fn ambient_dim(&self) -> usize {
        self.dim()
    }

    fn tangent_basis(&self, point: ArrayView1<'_, f64>) -> GeometryResult<Array2<f64>>;

    fn exp_map(
        &self,
        point: ArrayView1<'_, f64>,
        tangent_vec: ArrayView1<'_, f64>,
    ) -> GeometryResult<Array1<f64>>;

    fn log_map(
        &self,
        p_from: ArrayView1<'_, f64>,
        p_to: ArrayView1<'_, f64>,
    ) -> GeometryResult<Array1<f64>>;

    fn parallel_transport(
        &self,
        point_along: ArrayView2<'_, f64>,
        vec: ArrayView1<'_, f64>,
    ) -> GeometryResult<Array1<f64>>;

    fn metric_tensor(&self, point: ArrayView1<'_, f64>) -> GeometryResult<Array2<f64>>;

    fn christoffel_symbols(&self, point: ArrayView1<'_, f64>) -> GeometryResult<Vec<Array2<f64>>>;

    fn sectional_curvature(
        &self,
        point: ArrayView1<'_, f64>,
        tangent_pair: (ArrayView1<'_, f64>, ArrayView1<'_, f64>),
    ) -> GeometryResult<f64>;

    fn project_tangent(
        &self,
        point: ArrayView1<'_, f64>,
        vec: ArrayView1<'_, f64>,
    ) -> GeometryResult<Array1<f64>> {
        // Default projection is the identity (Euclidean-flat tangent space).
        // Validate the input point dimension matches the manifold's ambient
        // dimension so a caller passing the wrong-length vector fails fast
        // here rather than producing a silently mis-shaped tangent vector.
        let expected = self.ambient_dim();
        if point.len() != expected {
            return Err(GeometryError::DimensionMismatch {
                context: "project_tangent point",
                expected,
                got: point.len(),
            });
        }
        Ok(vec.to_owned())
    }

    fn retract(
        &self,
        point: ArrayView1<'_, f64>,
        tangent_vec: ArrayView1<'_, f64>,
    ) -> GeometryResult<Array1<f64>> {
        self.exp_map(point, tangent_vec)
    }

    /// Vector–Jacobian product of the ambient map `exp_p(v)`.
    ///
    /// Given a cotangent `grad_output` w.r.t. the ambient output of
    /// [`exp_map`](Self::exp_map), return `(grad_point, grad_tangent)`, the
    /// pullbacks w.r.t. the base point `p` and the (raw, unprojected) tangent
    /// input `v`. This is the analytic backward used by reverse-mode autodiff
    /// wrappers (e.g. the Python `torch.autograd.Function` around
    /// `manifold_exp_map`); it must never be the silent straight-through
    /// identity for a curved manifold.
    ///
    /// The default is the exact VJP for *flat* manifolds, where
    /// `exp_p(v) = p + v` in ambient coordinates and so both Jacobians are the
    /// identity (Euclidean, Circle, Torus, and products thereof). Curved
    /// manifolds **must** override this with their analytic Jacobi-field VJP;
    /// a manifold without a closed form must override it to return an error
    /// rather than inherit the wrong identity default.
    fn exp_map_vjp(
        &self,
        point: ArrayView1<'_, f64>,
        tangent_vec: ArrayView1<'_, f64>,
        grad_output: ArrayView1<'_, f64>,
    ) -> GeometryResult<(Array1<f64>, Array1<f64>)> {
        let m = self.ambient_dim();
        check_len("exp_map_vjp point", point.len(), m)?;
        check_len("exp_map_vjp tangent", tangent_vec.len(), m)?;
        check_len("exp_map_vjp grad_output", grad_output.len(), m)?;
        Ok((grad_output.to_owned(), grad_output.to_owned()))
    }
}

#[derive(Debug, Clone, PartialEq)]
pub enum ManifoldSpec {
    Euclidean(usize),
    Circle,
    Sphere { intrinsic_dim: usize },
    Torus { dim: usize },
    Grassmann { k: usize, n: usize },
    Stiefel { k: usize, n: usize },
    Spd { n: usize },
    Product(Vec<ManifoldSpec>),
}

impl ManifoldSpec {
    pub fn build(&self) -> Box<dyn RiemannianManifold> {
        match self {
            Self::Euclidean(dim) => Box::new(crate::geometry::EuclideanManifold::new(*dim)),
            Self::Circle => Box::new(crate::geometry::CircleManifold::new()),
            Self::Sphere { intrinsic_dim } => {
                Box::new(crate::geometry::SphereManifold::new(*intrinsic_dim))
            }
            Self::Torus { dim } => Box::new(crate::geometry::TorusManifold::new(*dim)),
            Self::Grassmann { k, n } => Box::new(crate::geometry::GrassmannManifold::new(*k, *n)),
            Self::Stiefel { k, n } => Box::new(crate::geometry::StiefelManifold::new(*k, *n)),
            Self::Spd { n } => Box::new(crate::geometry::SpdManifold::new(*n)),
            Self::Product(parts) => Box::new(crate::geometry::ProductManifold::new(
                parts.iter().map(Self::build).collect(),
            )),
        }
    }
}

pub(crate) const fn check_len(
    context: &'static str,
    got: usize,
    expected: usize,
) -> GeometryResult<()> {
    if got == expected {
        Ok(())
    } else {
        Err(GeometryError::DimensionMismatch {
            context,
            expected,
            got,
        })
    }
}

pub(crate) fn dot(a: ArrayView1<'_, f64>, b: ArrayView1<'_, f64>) -> f64 {
    assert_eq!(a.len(), b.len());
    let mut out = 0.0;
    for i in 0..a.len() {
        out += a[i] * b[i];
    }
    out
}

pub(crate) fn norm(a: ArrayView1<'_, f64>) -> f64 {
    dot(a, a).sqrt()
}

pub(crate) fn identity(n: usize) -> Array2<f64> {
    let mut out = Array2::<f64>::zeros((n, n));
    for i in 0..n {
        out[[i, i]] = 1.0;
    }
    out
}

pub(crate) fn zero_christoffel(dim: usize) -> Vec<Array2<f64>> {
    (0..dim).map(|_| Array2::<f64>::zeros((dim, dim))).collect()
}

pub(crate) fn wrap_angle(theta: f64) -> f64 {
    let two_pi = std::f64::consts::PI * 2.0;
    (theta + std::f64::consts::PI).rem_euclid(two_pi) - std::f64::consts::PI
}

pub(crate) fn sym(a: &Array2<f64>) -> Array2<f64> {
    let mut out = a.clone();
    for i in 0..a.nrows() {
        for j in 0..a.ncols() {
            out[[i, j]] = 0.5 * (a[[i, j]] + a[[j, i]]);
        }
    }
    out
}

pub(crate) fn from_flat(
    v: ArrayView1<'_, f64>,
    rows: usize,
    cols: usize,
) -> GeometryResult<Array2<f64>> {
    check_len("flat matrix", v.len(), rows * cols)?;
    let mut out = Array2::<f64>::zeros((rows, cols));
    for i in 0..rows {
        for j in 0..cols {
            out[[i, j]] = v[i * cols + j];
        }
    }
    Ok(out)
}

pub(crate) fn flatten(a: &Array2<f64>) -> Array1<f64> {
    let mut out = Array1::<f64>::zeros(a.nrows() * a.ncols());
    for i in 0..a.nrows() {
        for j in 0..a.ncols() {
            out[i * a.ncols() + j] = a[[i, j]];
        }
    }
    out
}

pub(crate) fn qr_thin(a: &Array2<f64>) -> (Array2<f64>, Array2<f64>) {
    let n = a.nrows();
    let k = a.ncols();
    let mut q = Array2::<f64>::zeros((n, k));
    let mut r = Array2::<f64>::zeros((k, k));
    for j in 0..k {
        let mut v = a.column(j).to_owned();
        for i in 0..j {
            let qi = q.column(i);
            let rij = dot(qi, v.view());
            r[[i, j]] = rij;
            for row in 0..n {
                v[row] -= rij * q[[row, i]];
            }
        }
        let nrm = norm(v.view());
        if nrm > GEOMETRY_EPS {
            r[[j, j]] = nrm;
            for row in 0..n {
                q[[row, j]] = v[row] / nrm;
            }
        } else if j < n {
            q[[j, j]] = 1.0;
            r[[j, j]] = 0.0;
        }
    }
    (q, r)
}

pub(crate) fn inverse(a: &Array2<f64>) -> GeometryResult<Array2<f64>> {
    let n = a.nrows();
    if n != a.ncols() {
        return Err(GeometryError::Singular("inverse requires a square matrix"));
    }
    let mut aug = Array2::<f64>::zeros((n, 2 * n));
    for i in 0..n {
        for j in 0..n {
            aug[[i, j]] = a[[i, j]];
        }
        aug[[i, n + i]] = 1.0;
    }
    for col in 0..n {
        let mut pivot = col;
        let mut best = aug[[col, col]].abs();
        for row in col + 1..n {
            let val = aug[[row, col]].abs();
            if val > best {
                best = val;
                pivot = row;
            }
        }
        if best < GEOMETRY_EPS {
            return Err(GeometryError::Singular("matrix inverse pivot underflow"));
        }
        if pivot != col {
            for j in 0..2 * n {
                let tmp = aug[[col, j]];
                aug[[col, j]] = aug[[pivot, j]];
                aug[[pivot, j]] = tmp;
            }
        }
        let scale = aug[[col, col]];
        for j in 0..2 * n {
            aug[[col, j]] /= scale;
        }
        for row in 0..n {
            if row == col {
                continue;
            }
            let factor = aug[[row, col]];
            for j in 0..2 * n {
                aug[[row, j]] -= factor * aug[[col, j]];
            }
        }
    }
    let mut out = Array2::<f64>::zeros((n, n));
    for i in 0..n {
        for j in 0..n {
            out[[i, j]] = aug[[i, n + j]];
        }
    }
    Ok(out)
}

pub(crate) fn jacobi_symmetric(a: &Array2<f64>) -> GeometryResult<(Array1<f64>, Array2<f64>)> {
    let n = a.nrows();
    if n != a.ncols() {
        return Err(GeometryError::InvalidPoint(
            "Jacobi eigensolver requires square input",
        ));
    }
    let mut d = sym(a);
    let mut v = identity(n);
    let max_iter = 64 * n.max(1) * n.max(1);
    // Relative convergence threshold: the largest off-diagonal magnitude must
    // fall to `1e-13 * ||A||_F`. A fixed absolute `1e-13` is meaningless for
    // matrices whose scale is far from unity (a well-scaled large-norm matrix
    // could never reach it; a tiny-norm matrix would "converge" trivially),
    // and silently returning the partially-diagonalized state after exhausting
    // `max_iter` hides genuine non-convergence (e.g. clustered/degenerate
    // spectra that stall the classical sweep). The Frobenius norm is invariant
    // under the orthogonal Jacobi rotations, so it is computed once from the
    // symmetrized input.
    let frob_norm = {
        let mut acc = 0.0;
        for i in 0..n {
            for j in 0..n {
                acc += d[[i, j]] * d[[i, j]];
            }
        }
        acc.sqrt()
    };
    let threshold = 1.0e-13 * frob_norm;
    let mut converged = false;
    for _ in 0..max_iter {
        let mut p = 0usize;
        let mut q = 0usize;
        let mut best = 0.0;
        for i in 0..n {
            for j in i + 1..n {
                let val = d[[i, j]].abs();
                if val > best {
                    best = val;
                    p = i;
                    q = j;
                }
            }
        }
        // `best <= threshold` (rather than `<`) makes the exactly-diagonal and
        // zero-norm cases (`best == threshold == 0`) converge immediately.
        if best <= threshold {
            converged = true;
            break;
        }
        let tau = (d[[q, q]] - d[[p, p]]) / (2.0 * d[[p, q]]);
        let t = tau.signum() / (tau.abs() + (1.0 + tau * tau).sqrt());
        let c = 1.0 / (1.0 + t * t).sqrt();
        let s = t * c;
        for k in 0..n {
            let dpk = d[[p, k]];
            let dqk = d[[q, k]];
            d[[p, k]] = c * dpk - s * dqk;
            d[[q, k]] = s * dpk + c * dqk;
        }
        for k in 0..n {
            let dkp = d[[k, p]];
            let dkq = d[[k, q]];
            d[[k, p]] = c * dkp - s * dkq;
            d[[k, q]] = s * dkp + c * dkq;
        }
        for k in 0..n {
            let vkp = v[[k, p]];
            let vkq = v[[k, q]];
            v[[k, p]] = c * vkp - s * vkq;
            v[[k, q]] = s * vkp + c * vkq;
        }
    }
    if !converged {
        return Err(GeometryError::Singular(
            "Jacobi eigensolver did not converge within max_iter (off-diagonal mass above 1e-13 * Frobenius norm)",
        ));
    }
    let mut evals = Array1::<f64>::zeros(n);
    for i in 0..n {
        evals[i] = d[[i, i]];
    }
    Ok((evals, v))
}

pub(crate) fn spectral_map_spd(
    a: &Array2<f64>,
    f: impl Fn(f64) -> GeometryResult<f64>,
) -> GeometryResult<Array2<f64>> {
    let (evals, evecs) = jacobi_symmetric(a)?;
    let n = a.nrows();
    let mut diag = Array2::<f64>::zeros((n, n));
    for i in 0..n {
        if evals[i] <= 0.0 || !evals[i].is_finite() {
            return Err(GeometryError::InvalidPoint(
                "SPD eigenvalue is not positive",
            ));
        }
        diag[[i, i]] = f(evals[i])?;
    }
    Ok(evecs.dot(&diag).dot(&evecs.t()))
}

pub(crate) fn spectral_map_symmetric(
    a: &Array2<f64>,
    f: impl Fn(f64) -> GeometryResult<f64>,
) -> GeometryResult<Array2<f64>> {
    let (evals, evecs) = jacobi_symmetric(a)?;
    let n = a.nrows();
    let mut diag = Array2::<f64>::zeros((n, n));
    for i in 0..n {
        diag[[i, i]] = f(evals[i])?;
    }
    Ok(evecs.dot(&diag).dot(&evecs.t()))
}

pub(crate) fn cholesky_spd(a: &Array2<f64>) -> GeometryResult<Array2<f64>> {
    let n = a.nrows();
    if n != a.ncols() {
        return Err(GeometryError::InvalidPoint(
            "Cholesky requires square input",
        ));
    }
    let mut l = Array2::<f64>::zeros((n, n));
    for i in 0..n {
        for j in 0..=i {
            let mut sum = a[[i, j]];
            for k in 0..j {
                sum -= l[[i, k]] * l[[j, k]];
            }
            if i == j {
                if sum <= GEOMETRY_EPS || !sum.is_finite() {
                    return Err(GeometryError::InvalidPoint(
                        "matrix is not positive definite",
                    ));
                }
                l[[i, j]] = sum.sqrt();
            } else {
                l[[i, j]] = sum / l[[j, j]];
            }
        }
    }
    Ok(l)
}

#[cfg(test)]
mod jacobi_tests {
    use super::{jacobi_symmetric, GeometryError};
    use ndarray::Array2;

    /// A large-norm SPD matrix has off-diagonal residuals after
    /// diagonalization that scale with `||A||_F`, so they sit far above the
    /// old *absolute* `1e-13` cutoff even when the decomposition is, in fact,
    /// fully converged. The relative threshold (`1e-13 * ||A||_F`) recognizes
    /// convergence here and returns the correct spectrum instead of grinding
    /// through `max_iter` sweeps and silently returning a partial diagonal.
    #[test]
    fn jacobi_converges_on_large_norm_spd() {
        // Q diag(1e8, 2e8, 3e8) Qᵀ for an orthogonal Q built from a planar
        // rotation in the (0,1) plane; eigenvalues are huge so the matrix
        // norm is ~1e8 and any absolute 1e-13 off-diagonal test is hopeless.
        let theta = 0.7_f64;
        let (c, s) = (theta.cos(), theta.sin());
        let mut q = Array2::<f64>::eye(3);
        q[[0, 0]] = c;
        q[[0, 1]] = -s;
        q[[1, 0]] = s;
        q[[1, 1]] = c;
        let lambda = [1.0e8_f64, 2.0e8, 3.0e8];
        let mut diag = Array2::<f64>::zeros((3, 3));
        for i in 0..3 {
            diag[[i, i]] = lambda[i];
        }
        let a = q.dot(&diag).dot(&q.t());

        let (evals, evecs) = jacobi_symmetric(&a).expect("large-norm SPD must converge");
        let mut sorted: Vec<f64> = evals.to_vec();
        sorted.sort_by(|x, y| x.partial_cmp(y).unwrap());
        for (got, want) in sorted.iter().zip(lambda.iter()) {
            assert!(
                (got - want).abs() <= 1.0e-6 * want,
                "eigenvalue mismatch: got {got}, want {want}"
            );
        }
        // V diag(evals) Vᵀ must reconstruct A (relative to its scale).
        let mut diag_e = Array2::<f64>::zeros((3, 3));
        for i in 0..3 {
            diag_e[[i, i]] = evals[i];
        }
        let recon = evecs.dot(&diag_e).dot(&evecs.t());
        for i in 0..3 {
            for j in 0..3 {
                assert!(
                    (recon[[i, j]] - a[[i, j]]).abs() <= 1.0e-6 * 3.0e8,
                    "reconstruction mismatch at ({i},{j})"
                );
            }
        }
    }

    /// A clustered/degenerate spectrum (two coincident eigenvalues) must still
    /// converge and reproduce the multiplicity. This guards against the
    /// relative threshold being so tight that ordinary near-degenerate SPD
    /// inputs trip the new non-convergence error.
    #[test]
    fn jacobi_handles_clustered_spectrum() {
        // diag(5, 5, 1) rotated in the (0,2) plane; the degenerate pair stays
        // degenerate under rotation.
        let theta = 0.4_f64;
        let (c, s) = (theta.cos(), theta.sin());
        let mut q = Array2::<f64>::eye(3);
        q[[0, 0]] = c;
        q[[0, 2]] = -s;
        q[[2, 0]] = s;
        q[[2, 2]] = c;
        let lambda = [5.0_f64, 5.0, 1.0];
        let mut diag = Array2::<f64>::zeros((3, 3));
        for i in 0..3 {
            diag[[i, i]] = lambda[i];
        }
        let a = q.dot(&diag).dot(&q.t());

        let (evals, evecs) = jacobi_symmetric(&a).expect("clustered SPD must converge");
        let mut sorted: Vec<f64> = evals.to_vec();
        sorted.sort_by(|x, y| x.partial_cmp(y).unwrap());
        assert!((sorted[0] - 1.0).abs() <= 1.0e-12);
        assert!((sorted[1] - 5.0).abs() <= 1.0e-12);
        assert!((sorted[2] - 5.0).abs() <= 1.0e-12);
        // Eigenvectors must remain orthonormal even across the degenerate pair.
        let gram = evecs.t().dot(&evecs);
        for i in 0..3 {
            for j in 0..3 {
                let want = if i == j { 1.0 } else { 0.0 };
                assert!(
                    (gram[[i, j]] - want).abs() <= 1.0e-12,
                    "eigenvectors not orthonormal at ({i},{j})"
                );
            }
        }
    }

    /// Non-convergence must now surface as `GeometryError::Singular` instead
    /// of a silently-returned partial diagonal. A symmetric input carrying a
    /// non-finite off-diagonal can never drive the largest off-diagonal
    /// magnitude below `1e-13 * ||A||_F` (the norm itself is non-finite), so
    /// the sweep exhausts `max_iter` and the solver must error rather than
    /// hand back the un-diagonalized matrix's diagonal.
    #[test]
    fn jacobi_errors_on_non_convergence() {
        let mut a = Array2::<f64>::eye(3);
        a[[0, 1]] = f64::NAN;
        a[[1, 0]] = f64::NAN;
        match jacobi_symmetric(&a) {
            Err(GeometryError::Singular(_)) => {}
            other => panic!("expected Singular non-convergence error, got {other:?}"),
        }
    }
}
