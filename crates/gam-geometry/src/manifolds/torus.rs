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

    // ── exp_map ───────────────────────────────────────────────────────────────

    #[test]
    fn exp_map_adds_tangent_to_point() {
        let m = TorusManifold::new(2);
        let p = array![0.1_f64, 0.2];
        let v = array![0.05_f64, -0.1];
        let q = m.exp_map(p.view(), v.view()).unwrap();
        assert!((q[0] - 0.15).abs() < 1e-12, "q[0]={}", q[0]);
        assert!((q[1] - 0.1).abs() < 1e-12, "q[1]={}", q[1]);
    }

    #[test]
    fn exp_map_wraps_angle_past_pi() {
        let m = TorusManifold::new(1);
        let p = array![3.0_f64];
        let v = array![0.5_f64];
        let q = m.exp_map(p.view(), v.view()).unwrap();
        // 3.5 > π, should wrap to 3.5 - 2π ≈ -2.783
        let expected = 3.5 - 2.0 * std::f64::consts::PI;
        assert!((q[0] - expected).abs() < 1e-12, "q[0]={}", q[0]);
    }

    #[test]
    fn exp_map_dimension_mismatch_is_error() {
        let m = TorusManifold::new(3);
        let p = array![0.1_f64, 0.2]; // wrong length
        let v = array![0.0_f64, 0.0, 0.0];
        assert!(m.exp_map(p.view(), v.view()).is_err());
    }

    // ── log_map ───────────────────────────────────────────────────────────────

    #[test]
    fn log_map_computes_wrapped_difference() {
        let m = TorusManifold::new(2);
        let p = array![0.1_f64, 0.5];
        let q = array![0.4_f64, 0.2];
        let v = m.log_map(p.view(), q.view()).unwrap();
        assert!((v[0] - 0.3).abs() < 1e-12, "v[0]={}", v[0]);
        assert!((v[1] - (-0.3)).abs() < 1e-12, "v[1]={}", v[1]);
    }

    #[test]
    fn exp_log_round_trip() {
        let m = TorusManifold::new(2);
        let p = array![0.2_f64, -0.5];
        let q = array![0.8_f64, 0.3];
        let v = m.log_map(p.view(), q.view()).unwrap();
        let recovered = m.exp_map(p.view(), v.view()).unwrap();
        assert!((recovered[0] - q[0]).abs() < 1e-12, "dim0={}", recovered[0]);
        assert!((recovered[1] - q[1]).abs() < 1e-12, "dim1={}", recovered[1]);
    }

    // ── parallel_transport ────────────────────────────────────────────────────

    #[test]
    fn parallel_transport_returns_vector_unchanged_flat_torus() {
        let m = TorusManifold::new(2);
        let path = array![[0.1_f64, 0.2], [0.3, 0.4]];
        let v = array![1.5_f64, -2.0];
        let transported = m.parallel_transport(path.view(), v.view()).unwrap();
        assert_eq!(transported.as_slice().unwrap(), v.as_slice().unwrap());
    }

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
