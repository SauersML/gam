use ndarray::{Array1, Array2, ArrayView1, ArrayView2};

use crate::geometry::manifold::{
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
