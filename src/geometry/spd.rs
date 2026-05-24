use ndarray::{Array1, Array2, ArrayView1, ArrayView2};

use crate::geometry::manifold::{
    GeometryError, GeometryResult, RiemannianManifold, check_len, cholesky_spd, dot, flatten,
    from_flat, inverse, spectral_map_spd, spectral_map_symmetric, sym,
};

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SpdManifold {
    n: usize,
}

impl SpdManifold {
    pub const fn new(n: usize) -> Self {
        Self { n }
    }

    fn matrix(&self, point: ArrayView1<'_, f64>) -> GeometryResult<Array2<f64>> {
        let p = sym(&from_flat(point, self.n, self.n)?);
        cholesky_spd(&p)?;
        Ok(p)
    }

    fn affine_inner(
        &self,
        p: &Array2<f64>,
        u: &Array2<f64>,
        v: &Array2<f64>,
    ) -> GeometryResult<f64> {
        let pinv = inverse(p)?;
        let a = pinv.dot(u).dot(&pinv).dot(v);
        let mut trace = 0.0;
        for i in 0..self.n {
            trace += a[[i, i]];
        }
        Ok(trace)
    }
}

impl RiemannianManifold for SpdManifold {
    fn dim(&self) -> usize {
        self.n * (self.n + 1) / 2
    }

    fn ambient_dim(&self) -> usize {
        self.n * self.n
    }

    fn tangent_basis(&self, point: ArrayView1<'_, f64>) -> GeometryResult<Array2<f64>> {
        check_len("SPD point", point.len(), self.ambient_dim())?;
        let mut out = Array2::<f64>::zeros((self.ambient_dim(), self.dim()));
        let mut col = 0usize;
        for i in 0..self.n {
            for j in i..self.n {
                out[[i * self.n + j, col]] = 1.0;
                if i != j {
                    out[[j * self.n + i, col]] = 1.0;
                }
                col += 1;
            }
        }
        Ok(out)
    }

    fn exp_map(
        &self,
        point: ArrayView1<'_, f64>,
        tangent_vec: ArrayView1<'_, f64>,
    ) -> GeometryResult<Array1<f64>> {
        let p = self.matrix(point)?;
        let u = sym(&from_flat(tangent_vec, self.n, self.n)?);
        let sqrt_p = spectral_map_spd(&p, |x| Ok(x.sqrt()))?;
        let inv_sqrt_p = spectral_map_spd(&p, |x| Ok(1.0 / x.sqrt()))?;
        let middle = inv_sqrt_p.dot(&u).dot(&inv_sqrt_p);
        let exp_middle = spectral_map_symmetric(&middle, |x| Ok(x.exp()))?;
        Ok(flatten(&sym(&sqrt_p.dot(&exp_middle).dot(&sqrt_p))))
    }

    fn log_map(
        &self,
        p_from: ArrayView1<'_, f64>,
        p_to: ArrayView1<'_, f64>,
    ) -> GeometryResult<Array1<f64>> {
        let p = self.matrix(p_from)?;
        let q = self.matrix(p_to)?;
        let sqrt_p = spectral_map_spd(&p, |x| Ok(x.sqrt()))?;
        let inv_sqrt_p = spectral_map_spd(&p, |x| Ok(1.0 / x.sqrt()))?;
        let middle = inv_sqrt_p.dot(&q).dot(&inv_sqrt_p);
        let log_middle = spectral_map_spd(&middle, |x| Ok(x.ln()))?;
        Ok(flatten(&sym(&sqrt_p.dot(&log_middle).dot(&sqrt_p))))
    }

    fn parallel_transport(
        &self,
        point_along: ArrayView2<'_, f64>,
        vec: ArrayView1<'_, f64>,
    ) -> GeometryResult<Array1<f64>> {
        check_len("SPD transported vector", vec.len(), self.ambient_dim())?;
        if point_along.nrows() < 2 {
            return Ok(flatten(&sym(&from_flat(vec, self.n, self.n)?)));
        }
        let p = self.matrix(point_along.row(0))?;
        let q = self.matrix(point_along.row(point_along.nrows() - 1))?;
        let u = sym(&from_flat(vec, self.n, self.n)?);
        let inv_sqrt_p = spectral_map_spd(&p, |x| Ok(1.0 / x.sqrt()))?;
        let middle = inv_sqrt_p.dot(&q).dot(&inv_sqrt_p);
        let e = spectral_map_spd(&middle, |x| Ok(x.sqrt()))?;
        let sqrt_p = spectral_map_spd(&p, |x| Ok(x.sqrt()))?;
        let a = sqrt_p.dot(&e).dot(&inv_sqrt_p);
        Ok(flatten(&sym(&a.dot(&u).dot(&a.t()))))
    }

    fn metric_tensor(&self, point: ArrayView1<'_, f64>) -> GeometryResult<Array2<f64>> {
        let p = self.matrix(point)?;
        let pinv = inverse(&p)?;
        let ambient = self.ambient_dim();
        let mut g = Array2::<f64>::zeros((ambient, ambient));
        for i in 0..self.n {
            for j in 0..self.n {
                for k in 0..self.n {
                    for l in 0..self.n {
                        g[[i * self.n + j, k * self.n + l]] = pinv[[i, k]] * pinv[[l, j]];
                    }
                }
            }
        }
        Ok(g)
    }

    fn christoffel_symbols(&self, point: ArrayView1<'_, f64>) -> GeometryResult<Vec<Array2<f64>>> {
        let p = self.matrix(point)?;
        let pinv = inverse(&p)?;
        let ambient = self.ambient_dim();
        let mut gamma = (0..ambient)
            .map(|_| Array2::<f64>::zeros((ambient, ambient)))
            .collect::<Vec<_>>();
        for a in 0..ambient {
            let ai = a / self.n;
            let aj = a % self.n;
            for b in 0..ambient {
                let bi = b / self.n;
                let bj = b % self.n;
                let mut u = Array2::<f64>::zeros((self.n, self.n));
                let mut v = Array2::<f64>::zeros((self.n, self.n));
                u[[ai, aj]] = 1.0;
                v[[bi, bj]] = 1.0;
                let c = -0.5 * (u.dot(&pinv).dot(&v) + v.dot(&pinv).dot(&u));
                for r in 0..self.n {
                    for s in 0..self.n {
                        gamma[r * self.n + s][[a, b]] = c[[r, s]];
                    }
                }
            }
        }
        Ok(gamma)
    }

    fn sectional_curvature(
        &self,
        point: ArrayView1<'_, f64>,
        tangent_pair: (ArrayView1<'_, f64>, ArrayView1<'_, f64>),
    ) -> GeometryResult<f64> {
        let p = self.matrix(point)?;
        let u = sym(&from_flat(tangent_pair.0, self.n, self.n)?);
        let v = sym(&from_flat(tangent_pair.1, self.n, self.n)?);
        let inv_sqrt_p = spectral_map_spd(&p, |x| Ok(1.0 / x.sqrt()))?;
        let a = inv_sqrt_p.dot(&u).dot(&inv_sqrt_p);
        let b = inv_sqrt_p.dot(&v).dot(&inv_sqrt_p);
        let comm = a.dot(&b) - b.dot(&a);
        let comm_norm = dot(flatten(&comm).view(), flatten(&comm).view());
        let uu = self.affine_inner(&p, &u, &u)?;
        let vv = self.affine_inner(&p, &v, &v)?;
        let uv = self.affine_inner(&p, &u, &v)?;
        let denom = uu * vv - uv * uv;
        if denom.abs() <= 1.0e-14 {
            return Err(GeometryError::Singular(
                "SPD sectional curvature plane is degenerate",
            ));
        }
        Ok(-0.25 * comm_norm / denom)
    }

    fn project_tangent(
        &self,
        point: ArrayView1<'_, f64>,
        vec: ArrayView1<'_, f64>,
    ) -> GeometryResult<Array1<f64>> {
        check_len("SPD projection point", point.len(), self.ambient_dim())?;
        Ok(flatten(&sym(&from_flat(vec, self.n, self.n)?)))
    }
}
