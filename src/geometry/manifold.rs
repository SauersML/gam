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
        _ = point;
        Ok(vec.to_owned())
    }

    fn retract(
        &self,
        point: ArrayView1<'_, f64>,
        tangent_vec: ArrayView1<'_, f64>,
    ) -> GeometryResult<Array1<f64>> {
        self.exp_map(point, tangent_vec)
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

pub(crate) fn check_len(context: &'static str, got: usize, expected: usize) -> GeometryResult<()> {
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
    debug_assert_eq!(a.len(), b.len());
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
        if best < 1.0e-13 {
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
