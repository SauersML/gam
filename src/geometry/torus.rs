use ndarray::{Array1, Array2, ArrayView1, ArrayView2};

use crate::geometry::manifold::{
    GeometryResult, RiemannianManifold, check_len, identity, wrap_angle, zero_christoffel,
};

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct TorusManifold {
    dim: usize,
}

impl TorusManifold {
    pub const fn new(dim: usize) -> Self {
        Self { dim }
    }
}

impl RiemannianManifold for TorusManifold {
    fn dim(&self) -> usize {
        self.dim
    }

    fn tangent_basis(&self, point: ArrayView1<'_, f64>) -> GeometryResult<Array2<f64>> {
        check_len("Torus point", point.len(), self.dim)?;
        Ok(identity(self.dim))
    }

    fn exp_map(
        &self,
        point: ArrayView1<'_, f64>,
        tangent_vec: ArrayView1<'_, f64>,
    ) -> GeometryResult<Array1<f64>> {
        check_len("Torus point", point.len(), self.dim)?;
        check_len("Torus tangent", tangent_vec.len(), self.dim)?;
        let mut out = Array1::<f64>::zeros(self.dim);
        for i in 0..self.dim {
            out[i] = wrap_angle(point[i] + tangent_vec[i]);
        }
        Ok(out)
    }

    fn log_map(
        &self,
        p_from: ArrayView1<'_, f64>,
        p_to: ArrayView1<'_, f64>,
    ) -> GeometryResult<Array1<f64>> {
        check_len("Torus source", p_from.len(), self.dim)?;
        check_len("Torus target", p_to.len(), self.dim)?;
        let mut out = Array1::<f64>::zeros(self.dim);
        for i in 0..self.dim {
            out[i] = wrap_angle(p_to[i] - p_from[i]);
        }
        Ok(out)
    }

    fn parallel_transport(
        &self,
        point_along: ArrayView2<'_, f64>,
        vec: ArrayView1<'_, f64>,
    ) -> GeometryResult<Array1<f64>> {
        if point_along.nrows() > 0 {
            check_len("Torus path width", point_along.ncols(), self.dim)?;
        }
        check_len("Torus transported vector", vec.len(), self.dim)?;
        Ok(vec.to_owned())
    }

    fn metric_tensor(&self, point: ArrayView1<'_, f64>) -> GeometryResult<Array2<f64>> {
        check_len("Torus metric point", point.len(), self.dim)?;
        Ok(identity(self.dim))
    }

    fn christoffel_symbols(&self, point: ArrayView1<'_, f64>) -> GeometryResult<Vec<Array2<f64>>> {
        check_len("Torus Christoffel point", point.len(), self.dim)?;
        Ok(zero_christoffel(self.dim))
    }

    fn sectional_curvature(
        &self,
        point: ArrayView1<'_, f64>,
        tangent_pair: (ArrayView1<'_, f64>, ArrayView1<'_, f64>),
    ) -> GeometryResult<f64> {
        check_len("Torus curvature point", point.len(), self.dim)?;
        check_len("Torus curvature tangent u", tangent_pair.0.len(), self.dim)?;
        check_len("Torus curvature tangent v", tangent_pair.1.len(), self.dim)?;
        Ok(0.0)
    }
}
