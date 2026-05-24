use ndarray::{Array1, Array2, ArrayView1, ArrayView2};

use crate::geometry::manifold::{
    GeometryResult, RiemannianManifold, check_len, dot, flatten, from_flat, identity, qr_thin, sym,
    zero_christoffel,
};

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct StiefelManifold {
    k: usize,
    n: usize,
}

impl StiefelManifold {
    pub const fn new(k: usize, n: usize) -> Self {
        Self { k, n }
    }

    fn qr_retraction(&self, y: &Array2<f64>) -> Array2<f64> {
        let (mut q, r) = qr_thin(y);
        for j in 0..self.k {
            if r[[j, j]] < 0.0 {
                for i in 0..self.n {
                    q[[i, j]] = -q[[i, j]];
                }
            }
        }
        q
    }
}

impl RiemannianManifold for StiefelManifold {
    fn dim(&self) -> usize {
        self.n * self.k - self.k * (self.k + 1) / 2
    }

    fn ambient_dim(&self) -> usize {
        self.n * self.k
    }

    fn tangent_basis(&self, point: ArrayView1<'_, f64>) -> GeometryResult<Array2<f64>> {
        check_len("Stiefel point", point.len(), self.ambient_dim())?;
        let mut columns: Vec<Array1<f64>> = Vec::with_capacity(self.dim());
        for col in 0..self.k {
            for row in 0..self.n {
                let mut e = Array2::<f64>::zeros((self.n, self.k));
                e[[row, col]] = 1.0;
                let mut v = self.project_tangent(point, flatten(&e).view())?;
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
                    return Ok(out);
                }
            }
        }
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
        Ok(flatten(&self.qr_retraction(&(y + tangent))))
    }

    fn log_map(
        &self,
        p_from: ArrayView1<'_, f64>,
        p_to: ArrayView1<'_, f64>,
    ) -> GeometryResult<Array1<f64>> {
        check_len("Stiefel source", p_from.len(), self.ambient_dim())?;
        check_len("Stiefel target", p_to.len(), self.ambient_dim())?;
        let diff = &p_to - &p_from;
        self.project_tangent(p_from, diff.view())
    }

    fn parallel_transport(
        &self,
        point_along: ArrayView2<'_, f64>,
        vec: ArrayView1<'_, f64>,
    ) -> GeometryResult<Array1<f64>> {
        check_len("Stiefel transported vector", vec.len(), self.ambient_dim())?;
        if point_along.nrows() == 0 {
            return Ok(vec.to_owned());
        }
        self.project_tangent(point_along.row(point_along.nrows() - 1), vec)
    }

    fn metric_tensor(&self, point: ArrayView1<'_, f64>) -> GeometryResult<Array2<f64>> {
        check_len("Stiefel metric point", point.len(), self.ambient_dim())?;
        Ok(identity(self.ambient_dim()))
    }

    fn christoffel_symbols(&self, point: ArrayView1<'_, f64>) -> GeometryResult<Vec<Array2<f64>>> {
        check_len("Stiefel Christoffel point", point.len(), self.ambient_dim())?;
        Ok(zero_christoffel(self.ambient_dim()))
    }

    fn sectional_curvature(
        &self,
        point: ArrayView1<'_, f64>,
        tangent_pair: (ArrayView1<'_, f64>, ArrayView1<'_, f64>),
    ) -> GeometryResult<f64> {
        check_len("Stiefel curvature point", point.len(), self.ambient_dim())?;
        check_len(
            "Stiefel curvature tangent u",
            tangent_pair.0.len(),
            self.ambient_dim(),
        )?;
        check_len(
            "Stiefel curvature tangent v",
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
        let correction = y.dot(&sym(&y.t().dot(&z)));
        Ok(flatten(&(z - correction)))
    }

    fn retract(
        &self,
        point: ArrayView1<'_, f64>,
        tangent_vec: ArrayView1<'_, f64>,
    ) -> GeometryResult<Array1<f64>> {
        self.exp_map(point, tangent_vec)
    }
}
