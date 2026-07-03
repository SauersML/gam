use ndarray::{Array1, Array2, ArrayView1, ArrayView2};

use crate::manifold::{
    GEOMETRY_EPS, GeometryError, GeometryResult, RiemannianManifold, check_len, dot, identity,
    zero_christoffel,
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

    /// Identity metric: the Riemannian gradient is the ambient gradient itself
    /// (the whole space is tangent). Overriding the metric-raising default keeps
    /// this O(d) instead of materializing the `d×d` identity basis and metric.
    fn riemannian_gradient(
        &self,
        point: ArrayView1<'_, f64>,
        euclidean_grad: ArrayView1<'_, f64>,
    ) -> GeometryResult<Array1<f64>> {
        self.project_tangent(point, euclidean_grad)
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
        // Sectional curvature lives on a 2-plane in the tangent space; a space of
        // dimension < 2 has no such plane, so the quantity is undefined rather
        // than flat-zero.
        if self.dim < 2 {
            return Err(GeometryError::Unsupported(
                "sectional curvature is undefined on a manifold of dimension below 2",
            ));
        }
        // Flat space has R ≡ 0, so K = 0 on any *nondegenerate* plane. But the
        // value 0/0 is undefined when the pair spans no plane: the squared
        // parallelogram area ‖u‖²‖v‖² − ⟨u,v⟩² must be nonzero.
        let uu = dot(tangent_pair.0, tangent_pair.0);
        let vv = dot(tangent_pair.1, tangent_pair.1);
        let uv = dot(tangent_pair.0, tangent_pair.1);
        let area_sq = uu * vv - uv * uv;
        if !area_sq.is_finite() || area_sq <= GEOMETRY_EPS {
            return Err(GeometryError::Singular(
                "sectional curvature undefined for collinear/degenerate tangent pair",
            ));
        }
        Ok(0.0)
    }
}

#[cfg(test)]
mod tests {
    use super::EuclideanManifold;
    use crate::manifold::{GeometryError, RiemannianManifold};
    use ndarray::array;

    // ── exp_map ───────────────────────────────────────────────────────────────

    #[test]
    fn exp_map_is_addition() {
        let m = EuclideanManifold::new(3);
        let p = array![1.0_f64, 2.0, 3.0];
        let v = array![0.5_f64, -1.0, 2.0];
        let q = m.exp_map(p.view(), v.view()).unwrap();
        assert!((q[0] - 1.5).abs() < 1e-14);
        assert!((q[1] - 1.0).abs() < 1e-14);
        assert!((q[2] - 5.0).abs() < 1e-14);
    }

    #[test]
    fn exp_map_dimension_mismatch_is_error() {
        let m = EuclideanManifold::new(3);
        let p = array![1.0_f64, 2.0]; // wrong length
        let v = array![0.0_f64, 0.0, 0.0];
        assert!(m.exp_map(p.view(), v.view()).is_err());
    }

    // ── log_map ───────────────────────────────────────────────────────────────

    #[test]
    fn log_map_is_subtraction() {
        let m = EuclideanManifold::new(2);
        let p = array![1.0_f64, 2.0];
        let q = array![4.0_f64, 0.0];
        let v = m.log_map(p.view(), q.view()).unwrap();
        assert!((v[0] - 3.0).abs() < 1e-14);
        assert!((v[1] - (-2.0)).abs() < 1e-14);
    }

    #[test]
    fn exp_log_round_trip() {
        let m = EuclideanManifold::new(3);
        let p = array![0.5_f64, -1.0, 2.0];
        let v = array![1.0_f64, 3.0, -0.5];
        let q = m.exp_map(p.view(), v.view()).unwrap();
        let v2 = m.log_map(p.view(), q.view()).unwrap();
        for i in 0..3 {
            assert!(
                (v2[i] - v[i]).abs() < 1e-14,
                "dim {i}: {} vs {}",
                v2[i],
                v[i]
            );
        }
    }

    // ── parallel_transport ────────────────────────────────────────────────────

    #[test]
    fn parallel_transport_returns_vector_unchanged() {
        let m = EuclideanManifold::new(3);
        let path = array![[0.0_f64, 0.0, 0.0], [1.0, 1.0, 1.0]];
        let v = array![2.0_f64, -3.0, 0.5];
        let result = m.parallel_transport(path.view(), v.view()).unwrap();
        for i in 0..3 {
            assert_eq!(result[i], v[i], "dim {i}");
        }
    }

    // ── metric_tensor ─────────────────────────────────────────────────────────

    #[test]
    fn metric_tensor_is_identity() {
        let m = EuclideanManifold::new(3);
        let p = array![1.0_f64, 2.0, 3.0];
        let g = m.metric_tensor(p.view()).unwrap();
        assert_eq!(g.dim(), (3, 3));
        for i in 0..3 {
            for j in 0..3 {
                let expected = if i == j { 1.0 } else { 0.0 };
                assert!((g[[i, j]] - expected).abs() < 1e-14);
            }
        }
    }

    #[test]
    fn sectional_curvature_is_zero_on_nondegenerate_plane() {
        let m = EuclideanManifold::new(2);
        let point = array![0.0, 0.0];
        let u = array![1.0, 0.0];
        let v = array![0.0, 1.0];
        let k = m
            .sectional_curvature(point.view(), (u.view(), v.view()))
            .expect("flat space has defined curvature on a nondegenerate plane");
        assert!(k.abs() < 1.0e-12, "expected 0, got {k}");
    }

    #[test]
    fn sectional_curvature_is_singular_for_collinear_pair() {
        let m = EuclideanManifold::new(2);
        let point = array![0.0, 0.0];
        let u = array![1.0, 0.0];
        // v parallel to u: zero parallelogram area.
        let v = array![3.0, 0.0];
        match m.sectional_curvature(point.view(), (u.view(), v.view())) {
            Err(GeometryError::Singular(_)) => {}
            other => panic!("expected Singular for collinear pair, got {other:?}"),
        }
    }

    #[test]
    fn sectional_curvature_is_unsupported_below_two_dimensions() {
        let m = EuclideanManifold::new(1);
        let point = array![0.0];
        let u = array![1.0];
        let v = array![1.0];
        match m.sectional_curvature(point.view(), (u.view(), v.view())) {
            Err(GeometryError::Unsupported(_)) => {}
            other => panic!("expected Unsupported in 1-D, got {other:?}"),
        }
    }
}
