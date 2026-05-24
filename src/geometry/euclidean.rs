use ndarray::{Array1, Array2, ArrayView1, ArrayView2};

use crate::geometry::manifold::{
    GeometryResult, RiemannianManifold, check_len, identity, zero_christoffel,
};

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct EuclideanManifold {
    dim: usize,
}

impl EuclideanManifold {
    pub const fn new(dim: usize) -> Self {
        Self { dim }
    }
}

impl RiemannianManifold for EuclideanManifold {
    fn dim(&self) -> usize {
        self.dim
    }

    fn tangent_basis(&self, point: ArrayView1<'_, f64>) -> GeometryResult<Array2<f64>> {
        check_len("Euclidean point", point.len(), self.dim)?;
        Ok(identity(self.dim))
    }

    fn exp_map(
        &self,
        point: ArrayView1<'_, f64>,
        tangent_vec: ArrayView1<'_, f64>,
    ) -> GeometryResult<Array1<f64>> {
        check_len("Euclidean point", point.len(), self.dim)?;
        check_len("Euclidean tangent", tangent_vec.len(), self.dim)?;
        Ok(&point + &tangent_vec)
    }

    fn log_map(
        &self,
        p_from: ArrayView1<'_, f64>,
        p_to: ArrayView1<'_, f64>,
    ) -> GeometryResult<Array1<f64>> {
        check_len("Euclidean source", p_from.len(), self.dim)?;
        check_len("Euclidean target", p_to.len(), self.dim)?;
        Ok(&p_to - &p_from)
    }

    fn parallel_transport(
        &self,
        point_along: ArrayView2<'_, f64>,
        vec: ArrayView1<'_, f64>,
    ) -> GeometryResult<Array1<f64>> {
        if point_along.nrows() > 0 {
            check_len("Euclidean path width", point_along.ncols(), self.dim)?;
        }
        check_len("Euclidean transported vector", vec.len(), self.dim)?;
        Ok(vec.to_owned())
    }

    fn metric_tensor(&self, point: ArrayView1<'_, f64>) -> GeometryResult<Array2<f64>> {
        check_len("Euclidean metric point", point.len(), self.dim)?;
        Ok(identity(self.dim))
    }

    fn christoffel_symbols(&self, point: ArrayView1<'_, f64>) -> GeometryResult<Vec<Array2<f64>>> {
        check_len("Euclidean Christoffel point", point.len(), self.dim)?;
        Ok(zero_christoffel(self.dim))
    }

    fn sectional_curvature(
        &self,
        point: ArrayView1<'_, f64>,
        tangent_pair: (ArrayView1<'_, f64>, ArrayView1<'_, f64>),
    ) -> GeometryResult<f64> {
        check_len("Euclidean curvature point", point.len(), self.dim)?;
        check_len(
            "Euclidean curvature tangent u",
            tangent_pair.0.len(),
            self.dim,
        )?;
        check_len(
            "Euclidean curvature tangent v",
            tangent_pair.1.len(),
            self.dim,
        )?;
        Ok(0.0)
    }
}
