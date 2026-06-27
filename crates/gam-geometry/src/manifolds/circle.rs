use ndarray::{Array1, Array2, ArrayView1, ArrayView2, array};

use crate::manifold::{
    GeometryError, GeometryResult, RiemannianManifold, check_len, identity, wrap_angle,
    zero_christoffel,
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

    /// Flat unit metric: the Riemannian gradient is the ambient gradient (the
    /// angular tangent line is the whole 1-D ambient space).
    fn riemannian_gradient(
        &self,
        point: ArrayView1<'_, f64>,
        euclidean_grad: ArrayView1<'_, f64>,
    ) -> GeometryResult<Array1<f64>> {
        self.project_tangent(point, euclidean_grad)
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
        // Sectional curvature is a function of a 2-plane in the tangent space:
        // its denominator ‖u‖²‖v‖² − ⟨u,v⟩² is the squared area of the tangent
        // parallelogram, which is identically 0 on a 1-D manifold (all tangents
        // are collinear). The quantity is therefore *undefined* here — returning
        // 0.0 would falsely report "flat", conflating the absence of a 2-plane
        // with vanishing curvature.
        Err(GeometryError::Unsupported(
            "sectional curvature is undefined on a 1-dimensional manifold",
        ))
    }
}

#[cfg(test)]
mod tests {
    use super::CircleManifold;
    use crate::manifold::{GeometryError, RiemannianManifold};
    use ndarray::array;

    #[test]
    fn sectional_curvature_is_unsupported_in_one_dimension() {
        let m = CircleManifold::new();
        let point = array![0.3];
        let u = array![1.0];
        let v = array![1.0];
        match m.sectional_curvature(point.view(), (u.view(), v.view())) {
            Err(GeometryError::Unsupported(_)) => {}
            other => panic!("expected Unsupported on 1-D circle, got {other:?}"),
        }
    }
}
