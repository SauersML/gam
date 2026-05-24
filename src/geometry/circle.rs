use ndarray::{Array1, Array2, ArrayView1, ArrayView2, array};

use crate::geometry::manifold::{
    GeometryResult, RiemannianManifold, check_len, identity, wrap_angle, zero_christoffel,
};

#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct CircleManifold;

impl CircleManifold {
    pub const fn new() -> Self {
        Self
    }
}

impl RiemannianManifold for CircleManifold {
    fn dim(&self) -> usize {
        1
    }

    fn tangent_basis(&self, point: ArrayView1<'_, f64>) -> GeometryResult<Array2<f64>> {
        check_len("Circle point", point.len(), 1)?;
        Ok(identity(1))
    }

    fn exp_map(
        &self,
        point: ArrayView1<'_, f64>,
        tangent_vec: ArrayView1<'_, f64>,
    ) -> GeometryResult<Array1<f64>> {
        check_len("Circle point", point.len(), 1)?;
        check_len("Circle tangent", tangent_vec.len(), 1)?;
        Ok(array![wrap_angle(point[0] + tangent_vec[0])])
    }

    fn log_map(
        &self,
        p_from: ArrayView1<'_, f64>,
        p_to: ArrayView1<'_, f64>,
    ) -> GeometryResult<Array1<f64>> {
        check_len("Circle source", p_from.len(), 1)?;
        check_len("Circle target", p_to.len(), 1)?;
        Ok(array![wrap_angle(p_to[0] - p_from[0])])
    }

    fn parallel_transport(
        &self,
        point_along: ArrayView2<'_, f64>,
        vec: ArrayView1<'_, f64>,
    ) -> GeometryResult<Array1<f64>> {
        if point_along.nrows() > 0 {
            check_len("Circle path width", point_along.ncols(), 1)?;
        }
        check_len("Circle transported vector", vec.len(), 1)?;
        Ok(vec.to_owned())
    }

    fn metric_tensor(&self, point: ArrayView1<'_, f64>) -> GeometryResult<Array2<f64>> {
        check_len("Circle metric point", point.len(), 1)?;
        Ok(identity(1))
    }

    fn christoffel_symbols(&self, point: ArrayView1<'_, f64>) -> GeometryResult<Vec<Array2<f64>>> {
        check_len("Circle Christoffel point", point.len(), 1)?;
        Ok(zero_christoffel(1))
    }

    fn sectional_curvature(
        &self,
        point: ArrayView1<'_, f64>,
        tangent_pair: (ArrayView1<'_, f64>, ArrayView1<'_, f64>),
    ) -> GeometryResult<f64> {
        check_len("Circle curvature point", point.len(), 1)?;
        check_len("Circle curvature tangent u", tangent_pair.0.len(), 1)?;
        check_len("Circle curvature tangent v", tangent_pair.1.len(), 1)?;
        Ok(0.0)
    }
}
