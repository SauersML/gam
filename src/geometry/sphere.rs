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
            return Err(GeometryError::InvalidPoint(
                "sphere normalization underflow",
            ));
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
        check_len(
            "Sphere curvature tangent u",
            tangent_pair.0.len(),
            self.ambient_dim(),
        )?;
        check_len(
            "Sphere curvature tangent v",
            tangent_pair.1.len(),
            self.ambient_dim(),
        )?;
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

    fn exp_map_vjp(
        &self,
        point: ArrayView1<'_, f64>,
        tangent_vec: ArrayView1<'_, f64>,
        grad_output: ArrayView1<'_, f64>,
    ) -> GeometryResult<(Array1<f64>, Array1<f64>)> {
        let m = self.ambient_dim();
        check_len("Sphere exp_map_vjp point", point.len(), m)?;
        check_len("Sphere exp_map_vjp tangent", tangent_vec.len(), m)?;
        check_len("Sphere exp_map_vjp grad", grad_output.len(), m)?;

        // Forward map: with `xi = (I - p p^T) v`, `theta = |xi|`,
        //   y = cos(theta) p + (sin(theta)/theta) xi.
        // We differentiate this closed form and return the transpose-applied
        // (vector–Jacobian) products w.r.t. the base point `p` and the raw
        // (unprojected) tangent input `v`.
        let c = dot(point, tangent_vec); // p · v
        let xi = &tangent_vec.to_owned() - &(point.to_owned() * c);
        let theta = norm(xi.view());
        let g = grad_output;
        let p = point;
        let v = tangent_vec;

        // Small-theta limit: y -> normalize(p + xi) and to first order the map
        // is exp_p(v) ≈ p + xi, so g_xi = (I - p p^T) g and the radial part
        // collapses. Use a Taylor-stable branch that matches `exp_map`'s
        // `theta < 1e-10` switch so backward is consistent with forward.
        if theta < 1.0e-10 {
            // xi ≈ 0. dy/dv = (I - p p^T); dy/dp = (1 - c) I - p v^T (from
            // y ≈ p + v - p(p·v), the first-order normalized expansion).
            let p_dot_g = dot(p, g.view());
            // grad_v = (I - p p^T) g = g - p (p·g).
            let grad_v = &g.to_owned() - &(p.to_owned() * p_dot_g);
            // grad_p = (1 - c) g - v (p·g)  [transpose of (1-c) I - p v^T].
            let grad_p = &(g.to_owned() * (1.0 - c)) - &(v.to_owned() * p_dot_g);
            return Ok((grad_p, grad_v));
        }

        let sin_t = theta.sin();
        let cos_t = theta.cos();
        let g_fn = sin_t / theta; // g(theta) = sin(theta)/theta
        // g'(theta) = (theta cos(theta) - sin(theta)) / theta^2.
        let g_prime = (theta * cos_t - sin_t) / (theta * theta);

        // We do NOT assume |p| == 1: the forward `exp_map` uses `point`
        // verbatim, so the honest VJP must be exact for any ambient `p`. With
        //   c = p·v,  n2 = |p|^2,  xi = v - c p,  theta = |xi|,
        //   y = cos(theta) p + g(theta) xi,
        // and using xi·p = c(1 - n2), the differentials give (see module
        // notes / derivation below) for any cotangent `g`:
        //   grad_v = alpha * w_v + g_fn (g - p (p·g)),
        //   grad_p = cos(theta) g + alpha * w_p - g_fn (c g + v (p·g)),
        // where
        //   alpha = -sin(theta)(p·g) + g'(theta)(xi·g),
        //   w_v   = (xi - c(1 - n2) p) / theta,
        //   w_p   = -(c xi + c(1 - n2) v) / theta.
        // For unit p (n2 == 1) the `c(1 - n2)` terms vanish and this reduces
        // to the textbook on-sphere Jacobi-field VJP.
        let n2 = dot(p, p);
        let p_dot_g = dot(p, g);
        let xi_dot_g = dot(xi.view(), g);
        let alpha = -sin_t * p_dot_g + g_prime * xi_dot_g;
        let cn = c * (1.0 - n2);

        // w_v = (xi - c(1-n2) p) / theta.
        let w_v = (&xi - &(p.to_owned() * cn)) / theta;
        let g_perp = &g.to_owned() - &(p.to_owned() * p_dot_g);
        let grad_v = &(&w_v * alpha) + &(&g_perp * g_fn);

        // w_p = -(c xi + c(1-n2) v) / theta.
        let w_p = (&(&xi * c) + &(v.to_owned() * cn)) / (-theta);
        let p_term = &(g.to_owned() * c) + &(v.to_owned() * p_dot_g);
        let grad_p = &(&(&w_p * alpha) + &(g.to_owned() * cos_t)) - &(&p_term * g_fn);

        Ok((grad_p, grad_v))
    }
}

fn validate_sphere_matrix(values: ArrayView2<'_, f64>) -> Result<(), String> {
    let (n, d) = values.dim();
    if n == 0 || d < 2 {
        return Err(
            "spherical values must have at least one row and at least two columns".to_string(),
        );
    }
    if let Some(((row, col), value)) = values.indexed_iter().find(|(_, v)| !v.is_finite()) {
        return Err(format!(
            "spherical values must contain only finite values; got {value} at ({row}, {col})"
        ));
    }
    Ok(())
}

fn normalize_sphere_matrix(values: ArrayView2<'_, f64>) -> Result<Array2<f64>, String> {
    validate_sphere_matrix(values)?;
    let (n, d) = values.dim();
    let mut out = Array2::<f64>::zeros((n, d));
    for row in 0..n {
        let row_norm = norm(values.row(row));
        if row_norm <= 0.0 {
            return Err("spherical rows must have non-zero norm".to_string());
        }
        for col in 0..d {
            out[[row, col]] = values[[row, col]] / row_norm;
        }
    }
    Ok(out)
}

fn normalize_weights(
    n: usize,
    weights: Option<ArrayView1<'_, f64>>,
) -> Result<Array1<f64>, String> {
    match weights {
        None => Ok(Array1::from_elem(n, 1.0 / n as f64)),
        Some(w) => {
            if w.len() != n {
                return Err("weights length must match the number of rows".to_string());
            }
            let mut total = 0.0_f64;
            for value in w.iter() {
                if !value.is_finite() || *value < 0.0 {
                    return Err(
                        "weights must be finite, non-negative, and have positive total".to_string(),
                    );
                }
                total += *value;
            }
            if total <= 0.0 {
                return Err(
                    "weights must be finite, non-negative, and have positive total".to_string(),
                );
            }
            Ok(w.mapv(|v| v / total))
        }
    }
}

fn sphere_orthogonal_unit(vector: ArrayView1<'_, f64>) -> Result<Array1<f64>, String> {
    let mut min_index = 0;
    let mut min_abs = vector[0].abs();
    for (index, value) in vector.iter().enumerate().skip(1) {
        let candidate = value.abs();
        if candidate < min_abs {
            min_abs = candidate;
            min_index = index;
        }
    }
    let axis_dot = vector[min_index];
    let mut tangent = Array1::<f64>::zeros(vector.len());
    tangent[min_index] = 1.0;
    for col in 0..vector.len() {
        tangent[col] -= axis_dot * vector[col];
    }
    let tangent_norm = norm(tangent.view());
    if tangent_norm <= 0.0 {
        return Err("cannot construct a tangent direction for the spherical mean".to_string());
    }
    Ok(tangent.mapv(|v| v / tangent_norm))
}

fn sphere_mean_candidates(
    values: ArrayView2<'_, f64>,
    weights: ArrayView1<'_, f64>,
) -> Result<Vec<Array1<f64>>, String> {
    let (n, d) = values.dim();
    let mut candidates: Vec<Array1<f64>> = Vec::new();
    let mut extrinsic = Array1::<f64>::zeros(d);
    for row in 0..n {
        for col in 0..d {
            extrinsic[col] += weights[row] * values[[row, col]];
        }
    }
    let ex_norm = norm(extrinsic.view());
    if ex_norm > 0.0 {
        candidates.push(extrinsic.mapv(|v| v / ex_norm));
    }
    let mut moment = Array2::<f64>::zeros((d, d));
    for row in 0..n {
        for r in 0..d {
            for c in 0..d {
                moment[[r, c]] += weights[row] * values[[row, r]] * values[[row, c]];
            }
        }
    }
    let mut v = Array1::<f64>::from_elem(d, 1.0 / (d as f64).sqrt());
    for _ in 0..64 {
        let mut nv = Array1::<f64>::zeros(d);
        for r in 0..d {
            let mut acc = 0.0;
            for c in 0..d {
                acc += moment[[r, c]] * v[c];
            }
            nv[r] = acc;
        }
        let nrm = norm(nv.view());
        if nrm <= 0.0 {
            break;
        }
        nv.mapv_inplace(|x| x / nrm);
        v = nv;
    }
    let v_norm = norm(v.view());
    if v_norm > 0.0 {
        let unit = v.mapv(|x| x / v_norm);
        candidates.push(unit.clone());
        candidates.push(unit.mapv(|x| -x));
    }
    Ok(candidates)
}

fn sphere_weighted_log_step(
    values: ArrayView2<'_, f64>,
    weights: ArrayView1<'_, f64>,
    base: ArrayView1<'_, f64>,
) -> Result<Array1<f64>, String> {
    let mut step = Array1::<f64>::zeros(base.len());
    for row in 0..values.nrows() {
        let mut dot_value = 0.0_f64;
        for col in 0..base.len() {
            dot_value += values[[row, col]] * base[col];
        }
        let dot_value = dot_value.clamp(-1.0, 1.0);
        if dot_value <= -1.0 + 1.0e-12 {
            return Err("spherical log map is undefined at antipodal points".to_string());
        }
        let theta = dot_value.acos();
        if theta < 1.0e-12 {
            continue;
        }
        let sin_theta = theta.sin();
        let scale = if sin_theta > 1.0e-12 {
            theta / sin_theta
        } else {
            1.0
        };
        for col in 0..base.len() {
            step[col] += weights[row] * (values[[row, col]] - dot_value * base[col]) * scale;
        }
    }
    Ok(step)
}

fn sphere_exp_single(
    tangent: ArrayView1<'_, f64>,
    base: ArrayView1<'_, f64>,
) -> Result<Array1<f64>, String> {
    let mut radial = 0.0_f64;
    for i in 0..base.len() {
        radial += tangent[i] * base[i];
    }
    let mut z = Array1::<f64>::zeros(base.len());
    for col in 0..base.len() {
        z[col] = tangent[col] - radial * base[col];
    }
    let r = norm(z.view());
    let mut out = Array1::<f64>::zeros(base.len());
    if r < 1.0e-12 {
        for col in 0..base.len() {
            out[col] = base[col] + z[col];
        }
    } else {
        let cos_r = r.cos();
        let sin_scale = r.sin() / r;
        for col in 0..base.len() {
            out[col] = cos_r * base[col] + sin_scale * z[col];
        }
    }
    let out_norm = norm(out.view());
    if !out_norm.is_finite() || out_norm <= 0.0 {
        return Err("spherical exponential map produced a non-finite point".to_string());
    }
    Ok(out.mapv(|v| v / out_norm))
}

fn sphere_frechet_objective(
    values: ArrayView2<'_, f64>,
    weights: ArrayView1<'_, f64>,
    base: ArrayView1<'_, f64>,
) -> f64 {
    let mut obj = 0.0_f64;
    for row in 0..values.nrows() {
        let mut dot_value = 0.0_f64;
        for col in 0..base.len() {
            dot_value += values[[row, col]] * base[col];
        }
        let dot_value = dot_value.clamp(-1.0, 1.0);
        let theta = dot_value.acos();
        obj += weights[row] * theta * theta;
    }
    obj
}

pub fn sphere_frechet_mean(
    points: ArrayView2<'_, f64>,
    weights: Option<ArrayView1<'_, f64>>,
    tol: f64,
    max_iter: usize,
) -> Result<Vec<f64>, String> {
    if !(tol.is_finite() && tol >= 0.0) {
        return Err("spherical Fréchet mean tolerance must be finite and non-negative".to_string());
    }
    let y = normalize_sphere_matrix(points)?;
    let w = normalize_weights(y.nrows(), weights)?;
    let mut candidates = sphere_mean_candidates(y.view(), w.view())?;
    if candidates.is_empty() {
        candidates.push(sphere_orthogonal_unit(y.row(0))?);
    }
    let mut best_mu: Option<Array1<f64>> = None;
    let mut best_obj = f64::INFINITY;
    for candidate in candidates {
        let mut mu = candidate;
        let mut failed = false;
        for _ in 0..max_iter {
            let step = match sphere_weighted_log_step(y.view(), w.view(), mu.view()) {
                Ok(step) => step,
                Err(_) => {
                    failed = true;
                    break;
                }
            };
            let step_norm = norm(step.view());
            if step_norm < tol {
                break;
            }
            mu = sphere_exp_single(step.view(), mu.view())?;
        }
        if failed {
            continue;
        }
        let obj = sphere_frechet_objective(y.view(), w.view(), mu.view());
        if obj < best_obj {
            best_obj = obj;
            best_mu = Some(mu);
        }
    }
    best_mu
        .map(|mu| mu.to_vec())
        .ok_or_else(|| "spherical Fréchet mean is not identifiable for these points".to_string())
}
