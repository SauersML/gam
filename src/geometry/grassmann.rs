use ndarray::{Array1, Array2, ArrayView1, ArrayView2};

use crate::geometry::manifold::{
    GEOMETRY_EPS, GeometryResult, RiemannianManifold, check_len, dot, flatten, from_flat, identity,
    inverse, jacobi_symmetric, qr_thin, zero_christoffel,
};

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct GrassmannManifold {
    k: usize,
    n: usize,
}

impl GrassmannManifold {
    pub const fn new(k: usize, n: usize) -> Self {
        Self { k, n }
    }

    fn orthonormalize(&self, y: &Array2<f64>) -> Array2<f64> {
        let (q, _) = qr_thin(y);
        q
    }

    fn compact_svd_from_tangent(
        &self,
        tangent: &Array2<f64>,
    ) -> GeometryResult<(Array2<f64>, Array1<f64>, Array2<f64>)> {
        let gram = tangent.t().dot(tangent);
        let (evals, v) = jacobi_symmetric(&gram)?;
        let mut sigma = Array1::<f64>::zeros(self.k);
        let mut u = Array2::<f64>::zeros((self.n, self.k));
        for j in 0..self.k {
            sigma[j] = evals[j].max(0.0).sqrt();
            if sigma[j] > GEOMETRY_EPS {
                let col = tangent.dot(&v.column(j).to_owned()) / sigma[j];
                for i in 0..self.n {
                    u[[i, j]] = col[i];
                }
            }
        }
        Ok((u, sigma, v))
    }
}

impl RiemannianManifold for GrassmannManifold {
    fn dim(&self) -> usize {
        self.k * (self.n - self.k)
    }

    fn ambient_dim(&self) -> usize {
        self.n * self.k
    }

    fn tangent_basis(&self, point: ArrayView1<'_, f64>) -> GeometryResult<Array2<f64>> {
        let y = from_flat(point, self.n, self.k)?;
        let mut columns: Vec<Array1<f64>> = Vec::with_capacity(self.dim());
        for col in 0..self.k {
            for row in 0..self.n {
                let mut e = Array2::<f64>::zeros((self.n, self.k));
                e[[row, col]] = 1.0;
                let p = self.project_tangent(point, flatten(&e).view())?;
                let mut v = p;
                for q in &columns {
                    let proj = dot(q.view(), v.view());
                    v -= &(q * proj);
                }
                let nrm = dot(v.view(), v.view()).sqrt();
                if nrm > 1.0e-10 {
                    columns.push(v / nrm);
                }
                if columns.len() == self.dim() {
                    let mut out = Array2::<f64>::zeros((self.ambient_dim(), self.dim()));
                    for j in 0..columns.len() {
                        for i in 0..self.ambient_dim() {
                            out[[i, j]] = columns[j][i];
                        }
                    }
                    drop(y);
                    return Ok(out);
                }
            }
        }
        drop(y);
        Ok(Array2::<f64>::zeros((self.ambient_dim(), columns.len())))
    }

    fn exp_map(
        &self,
        point: ArrayView1<'_, f64>,
        tangent_vec: ArrayView1<'_, f64>,
    ) -> GeometryResult<Array1<f64>> {
        let y = from_flat(point, self.n, self.k)?;
        let tangent = from_flat(
            self.project_tangent(point, tangent_vec)?.view(),
            self.n,
            self.k,
        )?;
        let (u, sigma, v) = self.compact_svd_from_tangent(&tangent)?;
        let mut cos_d = Array2::<f64>::zeros((self.k, self.k));
        let mut sin_d = Array2::<f64>::zeros((self.k, self.k));
        for i in 0..self.k {
            cos_d[[i, i]] = sigma[i].cos();
            sin_d[[i, i]] = sigma[i].sin();
        }
        let next = y.dot(&v).dot(&cos_d).dot(&v.t()) + u.dot(&sin_d).dot(&v.t());
        Ok(flatten(&self.orthonormalize(&next)))
    }

    fn log_map(
        &self,
        p_from: ArrayView1<'_, f64>,
        p_to: ArrayView1<'_, f64>,
    ) -> GeometryResult<Array1<f64>> {
        let y = from_flat(p_from, self.n, self.k)?;
        let z = from_flat(p_to, self.n, self.k)?;
        let yt_z = y.t().dot(&z);
        let inv = inverse(&yt_z)?;
        let normal = z - y.dot(&yt_z);
        let m = normal.dot(&inv);
        let gram = m.t().dot(&m);
        let (evals, v) = jacobi_symmetric(&gram)?;
        let mut sigma = Array1::<f64>::zeros(self.k);
        let mut u = Array2::<f64>::zeros((self.n, self.k));
        for j in 0..self.k {
            let tan_sigma = evals[j].max(0.0).sqrt();
            sigma[j] = tan_sigma.atan();
            if tan_sigma > GEOMETRY_EPS {
                let col = m.dot(&v.column(j).to_owned()) / tan_sigma;
                for i in 0..self.n {
                    u[[i, j]] = col[i];
                }
            }
        }
        let mut diag = Array2::<f64>::zeros((self.k, self.k));
        for i in 0..self.k {
            diag[[i, i]] = sigma[i];
        }
        Ok(flatten(&u.dot(&diag).dot(&v.t())))
    }

    fn parallel_transport(
        &self,
        point_along: ArrayView2<'_, f64>,
        vec: ArrayView1<'_, f64>,
    ) -> GeometryResult<Array1<f64>> {
        check_len(
            "Grassmann transported vector",
            vec.len(),
            self.ambient_dim(),
        )?;
        if point_along.nrows() == 0 {
            return Ok(vec.to_owned());
        }
        self.project_tangent(point_along.row(point_along.nrows() - 1), vec)
    }

    fn metric_tensor(&self, point: ArrayView1<'_, f64>) -> GeometryResult<Array2<f64>> {
        check_len("Grassmann metric point", point.len(), self.ambient_dim())?;
        Ok(identity(self.ambient_dim()))
    }

    fn christoffel_symbols(&self, point: ArrayView1<'_, f64>) -> GeometryResult<Vec<Array2<f64>>> {
        check_len(
            "Grassmann Christoffel point",
            point.len(),
            self.ambient_dim(),
        )?;
        Ok(zero_christoffel(self.ambient_dim()))
    }

    fn sectional_curvature(
        &self,
        point: ArrayView1<'_, f64>,
        tangent_pair: (ArrayView1<'_, f64>, ArrayView1<'_, f64>),
    ) -> GeometryResult<f64> {
        check_len("Grassmann curvature point", point.len(), self.ambient_dim())?;
        check_len(
            "Grassmann curvature tangent u",
            tangent_pair.0.len(),
            self.ambient_dim(),
        )?;
        check_len(
            "Grassmann curvature tangent v",
            tangent_pair.1.len(),
            self.ambient_dim(),
        )?;
        Ok(0.0)
    }

    fn project_tangent(
        &self,
        point: ArrayView1<'_, f64>,
        vec: ArrayView1<'_, f64>,
    ) -> GeometryResult<Array1<f64>> {
        let y = from_flat(point, self.n, self.k)?;
        let z = from_flat(vec, self.n, self.k)?;
        let yyt_z = y.dot(&y.t().dot(&z));
        let projected = z - yyt_z;
        Ok(flatten(&projected))
    }

    fn retract(
        &self,
        point: ArrayView1<'_, f64>,
        tangent_vec: ArrayView1<'_, f64>,
    ) -> GeometryResult<Array1<f64>> {
        let y = from_flat(point, self.n, self.k)?;
        let tangent = from_flat(
            self.project_tangent(point, tangent_vec)?.view(),
            self.n,
            self.k,
        )?;
        Ok(flatten(&self.orthonormalize(&(y + tangent))))
    }
}
