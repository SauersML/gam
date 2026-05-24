use ndarray::{Array1, Array2, ArrayView1, ArrayView2, s};

use crate::geometry::manifold::{
    GEOMETRY_EPS, GeometryError, GeometryResult, RiemannianManifold, check_len, dot, identity,
    norm, zero_christoffel,
};

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SphereManifold {
    intrinsic_dim: usize,
}

impl SphereManifold {
    pub const fn new(intrinsic_dim: usize) -> Self {
        Self { intrinsic_dim }
    }

    fn normalize(&self, x: Array1<f64>) -> GeometryResult<Array1<f64>> {
        let nrm = norm(x.view());
        if nrm <= GEOMETRY_EPS || !nrm.is_finite() {
            return Err(GeometryError::InvalidPoint("sphere normalization underflow"));
        }
        Ok(x / nrm)
    }
}

impl RiemannianManifold for SphereManifold {
    fn dim(&self) -> usize {
        self.intrinsic_dim
    }

    fn ambient_dim(&self) -> usize {
        self.intrinsic_dim + 1
    }

    fn tangent_basis(&self, point: ArrayView1<'_, f64>) -> GeometryResult<Array2<f64>> {
        let m = self.ambient_dim();
        check_len("Sphere point", point.len(), m)?;
        let mut anchor = 0usize;
        let mut max_abs = 0.0;
        for i in 0..m {
            if point[i].abs() > max_abs {
                max_abs = point[i].abs();
                anchor = i;
            }
        }
        let sign = if point[anchor] >= 0.0 { 1.0 } else { -1.0 };
        let mut u = point.to_owned() * sign;
        u[anchor] -= 1.0;
        let u_nrm = norm(u.view());
        let mut basis = Array2::<f64>::zeros((m, self.intrinsic_dim));
        if u_nrm <= GEOMETRY_EPS {
            let mut col = 0usize;
            for row in 0..m {
                if row != anchor {
                    basis[[row, col]] = 1.0;
                    col += 1;
                }
            }
            return Ok(basis);
        }
        u /= u_nrm;
        let mut col = 0usize;
        for j in 0..m {
            if j == anchor {
                continue;
            }
            let coef = 2.0 * u[j];
            for i in 0..m {
                basis[[i, col]] = -coef * u[i];
            }
            basis[[j, col]] += 1.0;
            col += 1;
        }
        Ok(basis)
    }

    fn exp_map(
        &self,
        point: ArrayView1<'_, f64>,
        tangent_vec: ArrayView1<'_, f64>,
    ) -> GeometryResult<Array1<f64>> {
        let m = self.ambient_dim();
        check_len("Sphere point", point.len(), m)?;
        check_len("Sphere tangent", tangent_vec.len(), m)?;
        let xi = self.project_tangent(point, tangent_vec)?;
        let theta = norm(xi.view());
        if theta < 1.0e-10 {
            return self.normalize(&point + &xi);
        }
        Ok(point.to_owned() * theta.cos() + xi * (theta.sin() / theta))
    }

    fn log_map(
        &self,
        p_from: ArrayView1<'_, f64>,
        p_to: ArrayView1<'_, f64>,
    ) -> GeometryResult<Array1<f64>> {
        let m = self.ambient_dim();
        check_len("Sphere source", p_from.len(), m)?;
        check_len("Sphere target", p_to.len(), m)?;
        let c = dot(p_from, p_to).clamp(-1.0, 1.0);
        let theta = c.acos();
        if theta < 1.0e-10 {
            return Ok(Array1::<f64>::zeros(m));
        }
        let mut u = &p_to - &(p_from.to_owned() * c);
        let u_nrm = norm(u.view());
        if u_nrm < 1.0e-10 {
            let basis = self.tangent_basis(p_from)?;
            return Ok(basis.slice(s![.., 0]).to_owned() * theta);
        }
        u *= theta / u_nrm;
        Ok(u)
    }

    fn parallel_transport(
        &self,
        point_along: ArrayView2<'_, f64>,
        vec: ArrayView1<'_, f64>,
    ) -> GeometryResult<Array1<f64>> {
        let m = self.ambient_dim();
        check_len("Sphere path width", point_along.ncols(), m)?;
        check_len("Sphere transported vector", vec.len(), m)?;
        if point_along.nrows() < 2 {
            return Ok(vec.to_owned());
        }
        let from = point_along.row(0);
        let to = point_along.row(point_along.nrows() - 1);
        let denom = 1.0 + dot(from, to);
        if denom.abs() < 1.0e-10 {
            return self.project_tangent(to, vec);
        }
        let scale = dot(vec, to) / denom;
        Ok(vec.to_owned() - &(from.to_owned() + to.to_owned()) * scale)
    }

    fn metric_tensor(&self, point: ArrayView1<'_, f64>) -> GeometryResult<Array2<f64>> {
        check_len("Sphere metric point", point.len(), self.ambient_dim())?;
        Ok(identity(self.ambient_dim()))
    }

    fn christoffel_symbols(&self, point: ArrayView1<'_, f64>) -> GeometryResult<Vec<Array2<f64>>> {
        check_len("Sphere Christoffel point", point.len(), self.ambient_dim())?;
        Ok(zero_christoffel(self.ambient_dim()))
    }

    fn sectional_curvature(
        &self,
        point: ArrayView1<'_, f64>,
        tangent_pair: (ArrayView1<'_, f64>, ArrayView1<'_, f64>),
    ) -> GeometryResult<f64> {
        check_len("Sphere curvature point", point.len(), self.ambient_dim())?;
        check_len("Sphere curvature tangent u", tangent_pair.0.len(), self.ambient_dim())?;
        check_len("Sphere curvature tangent v", tangent_pair.1.len(), self.ambient_dim())?;
        Ok(1.0)
    }

    fn project_tangent(
        &self,
        point: ArrayView1<'_, f64>,
        vec: ArrayView1<'_, f64>,
    ) -> GeometryResult<Array1<f64>> {
        check_len("Sphere projection point", point.len(), self.ambient_dim())?;
        check_len("Sphere projection vector", vec.len(), self.ambient_dim())?;
        Ok(vec.to_owned() - &(point.to_owned() * dot(point, vec)))
    }
}
