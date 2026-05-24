use ndarray::{Array1, Array2, ArrayView1, ArrayView2, s};

use crate::geometry::manifold::{GeometryResult, RiemannianManifold, check_len};

pub struct ProductManifold {
    components: Vec<Box<dyn RiemannianManifold>>,
}

impl ProductManifold {
    pub fn new(components: Vec<Box<dyn RiemannianManifold>>) -> Self {
        Self { components }
    }

    pub fn components(&self) -> &[Box<dyn RiemannianManifold>] {
        &self.components
    }
}

impl RiemannianManifold for ProductManifold {
    fn dim(&self) -> usize {
        self.components.iter().map(|c| c.dim()).sum()
    }

    fn ambient_dim(&self) -> usize {
        self.components.iter().map(|c| c.ambient_dim()).sum()
    }

    fn tangent_basis(&self, point: ArrayView1<'_, f64>) -> GeometryResult<Array2<f64>> {
        check_len("Product point", point.len(), self.ambient_dim())?;
        let mut out = Array2::<f64>::zeros((self.ambient_dim(), self.dim()));
        let mut row_off = 0usize;
        let mut col_off = 0usize;
        for component in &self.components {
            let m = component.ambient_dim();
            let d = component.dim();
            let q = component.tangent_basis(point.slice(s![row_off..row_off + m]))?;
            for i in 0..m {
                for j in 0..d {
                    out[[row_off + i, col_off + j]] = q[[i, j]];
                }
            }
            row_off += m;
            col_off += d;
        }
        Ok(out)
    }

    fn exp_map(
        &self,
        point: ArrayView1<'_, f64>,
        tangent_vec: ArrayView1<'_, f64>,
    ) -> GeometryResult<Array1<f64>> {
        check_len("Product point", point.len(), self.ambient_dim())?;
        check_len("Product tangent", tangent_vec.len(), self.ambient_dim())?;
        let mut out = Array1::<f64>::zeros(self.ambient_dim());
        let mut off = 0usize;
        for component in &self.components {
            let m = component.ambient_dim();
            let part = component.exp_map(
                point.slice(s![off..off + m]),
                tangent_vec.slice(s![off..off + m]),
            )?;
            for i in 0..m {
                out[off + i] = part[i];
            }
            off += m;
        }
        Ok(out)
    }

    fn log_map(
        &self,
        p_from: ArrayView1<'_, f64>,
        p_to: ArrayView1<'_, f64>,
    ) -> GeometryResult<Array1<f64>> {
        check_len("Product source", p_from.len(), self.ambient_dim())?;
        check_len("Product target", p_to.len(), self.ambient_dim())?;
        let mut out = Array1::<f64>::zeros(self.ambient_dim());
        let mut off = 0usize;
        for component in &self.components {
            let m = component.ambient_dim();
            let part = component.log_map(p_from.slice(s![off..off + m]), p_to.slice(s![off..off + m]))?;
            for i in 0..m {
                out[off + i] = part[i];
            }
            off += m;
        }
        Ok(out)
    }

    fn parallel_transport(
        &self,
        point_along: ArrayView2<'_, f64>,
        vec: ArrayView1<'_, f64>,
    ) -> GeometryResult<Array1<f64>> {
        check_len("Product path width", point_along.ncols(), self.ambient_dim())?;
        check_len("Product transported vector", vec.len(), self.ambient_dim())?;
        let mut out = Array1::<f64>::zeros(self.ambient_dim());
        let mut off = 0usize;
        for component in &self.components {
            let m = component.ambient_dim();
            let mut path = Array2::<f64>::zeros((point_along.nrows(), m));
            for row in 0..point_along.nrows() {
                for col in 0..m {
                    path[[row, col]] = point_along[[row, off + col]];
                }
            }
            let part = component.parallel_transport(path.view(), vec.slice(s![off..off + m]))?;
            for i in 0..m {
                out[off + i] = part[i];
            }
            off += m;
        }
        Ok(out)
    }

    fn metric_tensor(&self, point: ArrayView1<'_, f64>) -> GeometryResult<Array2<f64>> {
        check_len("Product metric point", point.len(), self.ambient_dim())?;
        let mut out = Array2::<f64>::zeros((self.ambient_dim(), self.ambient_dim()));
        let mut off = 0usize;
        for component in &self.components {
            let m = component.ambient_dim();
            let g = component.metric_tensor(point.slice(s![off..off + m]))?;
            for i in 0..m {
                for j in 0..m {
                    out[[off + i, off + j]] = g[[i, j]];
                }
            }
            off += m;
        }
        Ok(out)
    }

    fn christoffel_symbols(&self, point: ArrayView1<'_, f64>) -> GeometryResult<Vec<Array2<f64>>> {
        check_len("Product Christoffel point", point.len(), self.ambient_dim())?;
        let ambient = self.ambient_dim();
        let mut out = (0..ambient)
            .map(|_| Array2::<f64>::zeros((ambient, ambient)))
            .collect::<Vec<_>>();
        let mut off = 0usize;
        for component in &self.components {
            let m = component.ambient_dim();
            let gamma = component.christoffel_symbols(point.slice(s![off..off + m]))?;
            for k in 0..m {
                for i in 0..m {
                    for j in 0..m {
                        out[off + k][[off + i, off + j]] = gamma[k][[i, j]];
                    }
                }
            }
            off += m;
        }
        Ok(out)
    }

    fn sectional_curvature(
        &self,
        point: ArrayView1<'_, f64>,
        tangent_pair: (ArrayView1<'_, f64>, ArrayView1<'_, f64>),
    ) -> GeometryResult<f64> {
        check_len("Product curvature point", point.len(), self.ambient_dim())?;
        check_len("Product curvature tangent u", tangent_pair.0.len(), self.ambient_dim())?;
        check_len("Product curvature tangent v", tangent_pair.1.len(), self.ambient_dim())?;
        Ok(0.0)
    }

    fn project_tangent(
        &self,
        point: ArrayView1<'_, f64>,
        vec: ArrayView1<'_, f64>,
    ) -> GeometryResult<Array1<f64>> {
        check_len("Product projection point", point.len(), self.ambient_dim())?;
        check_len("Product projection vector", vec.len(), self.ambient_dim())?;
        let mut out = Array1::<f64>::zeros(self.ambient_dim());
        let mut off = 0usize;
        for component in &self.components {
            let m = component.ambient_dim();
            let part = component.project_tangent(point.slice(s![off..off + m]), vec.slice(s![off..off + m]))?;
            for i in 0..m {
                out[off + i] = part[i];
            }
            off += m;
        }
        Ok(out)
    }
}
