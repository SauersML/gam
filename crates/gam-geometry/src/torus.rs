use ndarray::{Array1, Array2, ArrayView1, ArrayView2};

use crate::manifold::{
    GEOMETRY_EPS, GeometryError, GeometryResult, RiemannianManifold, check_len, dot, identity,
    wrap_angle, zero_christoffel,
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

    /// Flat product metric: the Riemannian gradient is the ambient gradient (the
    /// per-angle tangent lines fill the whole ambient space).
    fn riemannian_gradient(
        &self,
        point: ArrayView1<'_, f64>,
        euclidean_grad: ArrayView1<'_, f64>,
    ) -> GeometryResult<Array1<f64>> {
        self.project_tangent(point, euclidean_grad)
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
        // Sectional curvature is defined only on a 2-plane in the tangent space.
        // A torus of dimension < 2 has no such plane (all tangents collinear),
        // so the quantity is undefined — returning 0.0 would falsely report
        // "flat" rather than "no 2-plane exists".
        if self.dim < 2 {
            return Err(GeometryError::Unsupported(
                "sectional curvature is undefined on a manifold of dimension below 2",
            ));
        }
        // For dim ≥ 2 the flat torus has identically zero curvature on any
        // *nondegenerate* tangent 2-plane. But the value is only defined when
        // (u, v) actually span a plane: the parallelogram area
        // ‖u‖²‖v‖² − ⟨u,v⟩² must be nonzero. A collinear (or zero) pair has zero
        // area and the sectional curvature 0/0 is undefined.
        let uu = dot(tangent_pair.0, tangent_pair.0);
        let vv = dot(tangent_pair.1, tangent_pair.1);
        let uv = dot(tangent_pair.0, tangent_pair.1);
        let area_sq = uu * vv - uv * uv;
        if area_sq <= GEOMETRY_EPS {
            return Err(GeometryError::Singular(
                "sectional curvature undefined for collinear/degenerate tangent pair",
            ));
        }
        Ok(0.0)
    }
}

#[cfg(test)]
mod tests {
    use super::TorusManifold;
    use crate::manifold::{GeometryError, RiemannianManifold};
    use ndarray::array;

    #[test]
    fn sectional_curvature_is_unsupported_below_two_dimensions() {
        let m = TorusManifold::new(1);
        let point = array![0.3];
        let u = array![1.0];
        let v = array![1.0];
        match m.sectional_curvature(point.view(), (u.view(), v.view())) {
            Err(GeometryError::Unsupported(_)) => {}
            other => panic!("expected Unsupported on 1-D torus, got {other:?}"),
        }
    }

    #[test]
    fn sectional_curvature_is_singular_for_collinear_pair() {
        let m = TorusManifold::new(2);
        let point = array![0.1, 0.2];
        let u = array![1.0, 0.0];
        // v parallel to u (collinear): zero parallelogram area.
        let v = array![2.0, 0.0];
        match m.sectional_curvature(point.view(), (u.view(), v.view())) {
            Err(GeometryError::Singular(_)) => {}
            other => panic!("expected Singular for collinear pair, got {other:?}"),
        }
    }

    #[test]
    fn sectional_curvature_is_zero_for_independent_pair() {
        let m = TorusManifold::new(2);
        let point = array![0.1, 0.2];
        let u = array![1.0, 0.0];
        let v = array![0.0, 1.0];
        let k = m
            .sectional_curvature(point.view(), (u.view(), v.view()))
            .expect("flat torus has defined curvature on a nondegenerate plane");
        assert!(
            k.abs() <= 1.0e-12,
            "flat torus sectional curvature must be 0, got {k}"
        );
    }
}
