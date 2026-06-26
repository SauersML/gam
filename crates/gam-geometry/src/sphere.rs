use ndarray::{Array1, Array2, ArrayView1, ArrayView2};

use crate::manifold::{
    GEOMETRY_EPS, GeometryError, GeometryResult, RiemannianManifold, check_len, dot, identity, norm,
};
use crate::normalize_weights;

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SphereManifold {
    intrinsic_dim: usize,
}

impl SphereManifold {
    /// Tolerance on `‖p‖² − 1` for accepting a point as on the unit sphere.
    const UNIT_TOL: f64 = 1.0e-6;

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

    /// Reject base points that are not on the unit sphere. This guards the maps
    /// that are only meaningful at a genuine manifold point — `log_map`,
    /// `metric_tensor`, `parallel_transport`, `tangent_basis`, `project_tangent`
    /// — where a non-unit `p` makes `v − p(pᵀv)` not even tangent. It is
    /// deliberately *not* applied to `exp_map` / `exp_map_vjp`, which are the
    /// honest-ambient forward/adjoint pair used by reverse-mode autodiff: they
    /// consume `point` verbatim so finite-difference probes can step off the
    /// sphere (see `exp_map`). The tolerance is loose enough to absorb the float
    /// drift of on-manifold iterates (retraction renormalizes to ~1e-15) yet
    /// still rejects genuinely off-manifold inputs.
    fn require_unit(&self, point: ArrayView1<'_, f64>) -> GeometryResult<()> {
        let n2 = dot(point, point);
        if !n2.is_finite() || (n2 - 1.0).abs() > Self::UNIT_TOL {
            return Err(GeometryError::InvalidPoint(
                "sphere operation requires a unit-norm base point",
            ));
        }
        Ok(())
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
        self.require_unit(point)?;
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
        // Honest-ambient exponential: `point` is used verbatim and is NOT
        // required to satisfy ‖p‖ = 1. This is the forward whose reverse mode
        // [`exp_map_vjp`](Self::exp_map_vjp) differentiates, and the
        // finite-difference pins (`tests/sphere_exp_map_vjp_matches_finite_difference.rs`)
        // step `point` off the unit sphere on purpose to exercise the general
        // `|p|² ≠ 1` branch — so this map must stay smooth in the raw ambient
        // coordinates rather than reject them. With `c = p·v`, the tangent
        // component is `xi = v − c·p` (the orthogonal projection only when
        // ‖p‖ = 1); `require_unit` is reserved for the maps that genuinely need
        // a manifold point (`log_map`, `metric_tensor`, `parallel_transport`).
        let c = dot(point, tangent_vec);
        let xi = &tangent_vec.to_owned() - &(point.to_owned() * c);
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
        self.require_unit(p_from)?;
        self.require_unit(p_to)?;
        let c = dot(p_from, p_to).clamp(-1.0, 1.0);
        // Geodesic length via the chord/haversine form theta = 2·arcsin(|p-q|/2)
        // rather than acos(p·q). For nearby unit vectors p·q = 1 − |p-q|²/2
        // saturates to ~1, so `1 − c` carries only ~eps/(theta²/2) relative
        // accuracy and acos(1−x) ≈ √(2x) amplifies it to ~eps/theta² error in
        // theta. The chord |p-q| is formed straight from the coordinates (no
        // near-1 subtraction) and stays accurate to ~1e-12 across the range.
        // Identical points give chord 0 → theta 0 (preserving the short-circuit
        // below); the dot product c is still used for the tangent direction.
        let mut chord_sq = 0.0_f64;
        for i in 0..m {
            let d = p_to[i] - p_from[i];
            chord_sq += d * d;
        }
        let theta = 2.0 * (0.5 * chord_sq.sqrt()).min(1.0).asin();
        if theta < 1.0e-10 {
            return Ok(Array1::<f64>::zeros(m));
        }
        let mut u = &p_to - &(p_from.to_owned() * c);
        let u_nrm = norm(u.view());
        if u_nrm < 1.0e-10 {
            // theta ≈ π with a vanishing tangent direction means p_to is the
            // antipode of p_from. The logarithm there is multivalued — every
            // unit u ⟂ p_from satisfies Exp_{p_from}(πu) = −p_from — so there
            // is no single correct answer to return. Surface it rather than
            // fabricating an arbitrary basis direction (which was also
            // discontinuous across the cut locus).
            return Err(GeometryError::Singular(
                "sphere log map is undefined at the antipode (cut locus)",
            ));
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
        self.require_unit(from)?;
        self.require_unit(to)?;
        let denom = 1.0 + dot(from, to);
        if denom.abs() < 1.0e-10 {
            // from ≈ −to: parallel transport across the cut locus depends on
            // which geodesic is chosen (transporting along the great circle
            // through e₂ versus e₃ gives different results), so with only the
            // endpoints there is no well-defined answer. The previous fallback
            // merely projected `vec` into T_to S, which is not parallel
            // transport. Require the caller to supply an actual path instead.
            return Err(GeometryError::Singular(
                "sphere parallel transport across antipodal endpoints is path-dependent",
            ));
        }
        let scale = dot(vec, to) / denom;
        Ok(vec.to_owned() - &(from.to_owned() + to.to_owned()) * scale)
    }

    fn metric_tensor(&self, point: ArrayView1<'_, f64>) -> GeometryResult<Array2<f64>> {
        check_len("Sphere metric point", point.len(), self.ambient_dim())?;
        self.require_unit(point)?;
        Ok(identity(self.ambient_dim()))
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
        // Sectional curvature is the curvature of a *2-plane* in the tangent
        // space. The intrinsic tangent space of S^d has dimension d, so no such
        // plane exists for d < 2 (S^1 is a single tangent line) — returning the
        // constant +1 would assert a curvature on a plane that does not exist.
        if self.dim() < 2 {
            return Err(GeometryError::Unsupported(
                "sectional curvature is undefined on a sphere of dimension below 2",
            ));
        }
        // A non-unit base point has no well-defined tangent projection, so the
        // 2-plane is meaningless; reject before projecting.
        self.require_unit(point)?;
        // K(u, v) = ⟨R(u,v)v, u⟩ / (‖u‖²‖v‖² − ⟨u,v⟩²). On the unit sphere the
        // numerator equals the squared parallelogram area of the *tangential*
        // components, so K = +1 — but only when that area is nonzero. A zero,
        // collinear, or purely-radial pair gives 0/0, which is undefined, not 1.
        // Strip the radial component so the area is computed on genuine tangent
        // vectors (a pair that is collinear only after projection still spans no
        // tangent plane).
        let pu = dot(point, tangent_pair.0);
        let pv = dot(point, tangent_pair.1);
        let u_t = tangent_pair.0.to_owned() - &(point.to_owned() * pu);
        let v_t = tangent_pair.1.to_owned() - &(point.to_owned() * pv);
        let uu = dot(u_t.view(), u_t.view());
        let vv = dot(v_t.view(), v_t.view());
        let uv = dot(u_t.view(), v_t.view());
        let area_sq = uu * vv - uv * uv;
        if !area_sq.is_finite() || area_sq <= GEOMETRY_EPS {
            return Err(GeometryError::Singular(
                "sectional curvature undefined for collinear/degenerate tangent pair",
            ));
        }
        Ok(1.0)
    }

    fn project_tangent(
        &self,
        point: ArrayView1<'_, f64>,
        vec: ArrayView1<'_, f64>,
    ) -> GeometryResult<Array1<f64>> {
        check_len("Sphere projection point", point.len(), self.ambient_dim())?;
        check_len("Sphere projection vector", vec.len(), self.ambient_dim())?;
        self.require_unit(point)?;
        Ok(vec.to_owned() - &(point.to_owned() * dot(point, vec)))
    }

    /// The round sphere carries the metric *induced* from the ambient Euclidean
    /// inner product, so the Riemannian gradient is the orthogonal projection of
    /// the ambient gradient onto the tangent space `T_pS = p^⊥` — exactly
    /// [`project_tangent`]. (The metric-raising default would give the same
    /// vector but only after building the dense `m×m` identity metric.)
    fn riemannian_gradient(
        &self,
        point: ArrayView1<'_, f64>,
        euclidean_grad: ArrayView1<'_, f64>,
    ) -> GeometryResult<Array1<f64>> {
        self.project_tangent(point, euclidean_grad)
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
        // No `require_unit` here: this is the exact adjoint of the honest-ambient
        // [`exp_map`](Self::exp_map), which uses `point` verbatim. The general
        // branch below carries the `c(1 − |p|²)` terms, so it is correct for any
        // ambient `p`; gating on ‖p‖ = 1 would make that branch unreachable and
        // break the off-sphere finite-difference contract the VJP exists for.

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

pub fn validate_sphere_matrix(values: ArrayView2<'_, f64>) -> Result<(), String> {
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

pub fn normalize_sphere_matrix(values: ArrayView2<'_, f64>) -> Result<Array2<f64>, String> {
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

/// Batched Riemannian log map of each row of `values` at a single `base`, in
/// ambient tangent coordinates. Inputs are normalized onto the unit sphere
/// first (unlike the strict [`SphereManifold::log_map`] trait method, which
/// requires unit inputs); the geodesic angle uses the numerically stable
/// `atan2(|u|, p·q)` form. Errors at antipodal points. This is the
/// response-geometry companion to the trait method.
pub fn response_sphere_log_map(
    values: ArrayView2<'_, f64>,
    base: ArrayView1<'_, f64>,
) -> Result<Array2<f64>, String> {
    let y = normalize_sphere_matrix(values)?;
    let base2 = Array2::from_shape_fn((1, base.len()), |(_, j)| base[j]);
    let b_mat = normalize_sphere_matrix(base2.view())?;
    let (n, d) = y.dim();
    if d != b_mat.ncols() {
        return Err("spherical values and base point have different dimensions".to_string());
    }
    // The per-row geodesic angle needs the inner product `pᵢ·base` for every one
    // of the n rows. Collected together this is `Y · base` (n×d · d → n); cast as
    // the n×d · d×1 product it row-tiles across ALL GPUs (each device handles its
    // observation-row tile with `base` broadcast), falling back to the
    // single-device shim for small batches. The remaining per-row scalar work
    // (atan2 angle, tangent scaling) is identical to the elementwise form.
    // f64 throughout.
    let base_col = b_mat.slice(ndarray::s![0..1, ..]).t().to_owned();
    let dots_mat = crate::manifold::fast_ab_rows_multi_gpu(y.view(), base_col.view());
    let dots = dots_mat.column(0).to_owned();
    let mut out = Array2::<f64>::zeros((n, d));
    for row in 0..n {
        let mut dot = dots[row];
        dot = dot.clamp(-1.0, 1.0);
        if dot <= -1.0 + 1.0e-12 {
            return Err("spherical log map is undefined at antipodal points".to_string());
        }
        // Geodesic angle via theta = atan2(|u|, p·q) with u = q − (p·q)p, the
        // component of q orthogonal to p (|u| = sin theta). For nearby points
        // p·q rounds to exactly 1.0 in f64 and acos(p·q) collapses a genuine
        // ~1e-9 distance to 0; |u| is formed straight from the coordinates with
        // no near-1 subtraction, so atan2(|u|, p·q) ≈ |u| stays accurate and
        // the tangent norm equals the geodesic distance as documented.
        let mut s_sq = 0.0_f64;
        for col in 0..d {
            let uc = y[[row, col]] - dot * b_mat[[0, col]];
            s_sq += uc * uc;
        }
        let s = s_sq.sqrt();
        if s < 1.0e-12 {
            for col in 0..d {
                out[[row, col]] = 0.0;
            }
        } else {
            let scale = s.atan2(dot) / s;
            for col in 0..d {
                out[[row, col]] = (y[[row, col]] - dot * b_mat[[0, col]]) * scale;
            }
        }
    }
    Ok(out)
}

/// Batched Riemannian exp map of each tangent row at a single `base`, returning
/// points on the unit sphere. The base is normalized first; the orthogonal
/// component of the tangent drives the geodesic step `cos(r)·p + (sin r / r)·z`.
/// This is the response-geometry companion to [`SphereManifold::exp_map`].
pub fn response_sphere_exp_map(
    tangent: ArrayView2<'_, f64>,
    base: ArrayView1<'_, f64>,
) -> Result<Array2<f64>, String> {
    let base2 = Array2::from_shape_fn((1, base.len()), |(_, j)| base[j]);
    let b_mat = normalize_sphere_matrix(base2.view())?;
    let (n, d) = tangent.dim();
    if d != b_mat.ncols() {
        return Err("spherical tangent and base point have different dimensions".to_string());
    }
    if !tangent.iter().all(|v| v.is_finite()) {
        return Err("spherical tangent must contain only finite values".to_string());
    }
    // The radial component `tangentᵢ·base` for every row is `T · base`
    // (n×d · d → n); cast as the n×d · d×1 product it row-tiles across ALL GPUs
    // (per-observation-row tiles, `base` broadcast), with a single-device
    // fallback. The per-row geodesic step that follows is identical scalar math.
    // f64 throughout.
    let base_col = b_mat.slice(ndarray::s![0..1, ..]).t().to_owned();
    let radials_mat = crate::manifold::fast_ab_rows_multi_gpu(tangent, base_col.view());
    let radials = radials_mat.column(0).to_owned();
    let mut out = Array2::<f64>::zeros((n, d));
    for row in 0..n {
        let radial = radials[row];
        let mut z = vec![0.0_f64; d];
        let mut r_sq = 0.0_f64;
        for col in 0..d {
            let v = tangent[[row, col]] - radial * b_mat[[0, col]];
            z[col] = v;
            r_sq += v * v;
        }
        let r = r_sq.sqrt();
        let mut norm_sq = 0.0_f64;
        if r < 1.0e-12 {
            for col in 0..d {
                let v = b_mat[[0, col]] + z[col];
                out[[row, col]] = v;
                norm_sq += v * v;
            }
        } else {
            let cos_r = r.cos();
            let sin_scale = r.sin() / r;
            for col in 0..d {
                let v = cos_r * b_mat[[0, col]] + sin_scale * z[col];
                out[[row, col]] = v;
                norm_sq += v * v;
            }
        }
        let norm = norm_sq.sqrt();
        if !norm.is_finite() || norm <= 0.0 {
            return Err("spherical exponential map produced a non-finite point".to_string());
        }
        for col in 0..d {
            out[[row, col]] /= norm;
        }
    }
    Ok(out)
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

/// Fixed power-iteration step count used when seeding the spherical-mean search
/// with the dominant axis of the weighted second-moment matrix. The seed only
/// needs to land in the right basin (the subsequent Riemannian iteration
/// refines it), so a modest fixed budget suffices and avoids a per-call
/// convergence test on the hot seeding path.
const SPHERE_SEED_POWER_ITERS: usize = 64;

/// Power-iteration step count for the standalone dominant-axis helper. Larger
/// than the seed budget because its result is consumed directly (not refined
/// downstream), so it iterates further toward the true leading eigenvector.
const SPHERE_DOMINANT_AXIS_POWER_ITERS: usize = 128;

fn sphere_mean_candidates(
    values: ArrayView2<'_, f64>,
    weights: ArrayView1<'_, f64>,
) -> Result<Vec<Array1<f64>>, String> {
    let (_, d) = values.dim();
    let mut candidates: Vec<Array1<f64>> = Vec::new();
    // Weighted extrinsic mean `Σ wᵢ pᵢ = Pᵀ w`: a single matrix–vector product
    // over all points, dispatched to GPU by `fast_atv` for large batches.
    let extrinsic = gam_linalg::faer_ndarray::fast_atv(&values, &weights);
    let ex_norm = norm(extrinsic.view());
    if ex_norm > 0.0 {
        candidates.push(extrinsic.mapv(|v| v / ex_norm));
    }
    // `M = Σ wᵢ pᵢ pᵢᵀ = Pᵀ diag(w) P` over all n points: the same GPU-dispatched
    // weighted cross-product used by `sphere_second_moment`.
    let moment = sphere_second_moment(values, weights);
    let mut v = Array1::<f64>::from_elem(d, 1.0 / (d as f64).sqrt());
    for _ in 0..SPHERE_SEED_POWER_ITERS {
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
    // The dominant eigenvector of `M` always lies IN the subspace where the data
    // is spread. When the points are balanced around a great circle (e.g. an
    // equilateral triangle on the equator, `M = diag(1.5, 1.5, 0)`), the extrinsic
    // mean is the zero vector (dropped above) and every dominant-eigenvector seed
    // sits on the equator, so descent converges to an equatorial data point. The
    // TRUE Fréchet mean is on the axis ORTHOGONAL to the spread — the SMALLEST
    // eigenvalue eigenvector. Seed the descent from the full orthonormal eigenbasis
    // of `M` (both signs); the caller keeps the lowest-objective converged result,
    // so non-degenerate inputs are unaffected.
    for axis in sphere_eigenbasis(moment.view()) {
        let nrm = norm(axis.view());
        if nrm > 0.0 {
            let unit = axis.mapv(|x| x / nrm);
            candidates.push(unit.clone());
            candidates.push(unit.mapv(|x| -x));
        }
    }
    Ok(candidates)
}

/// Orthonormal eigenbasis of a symmetric PSD matrix via power iteration with
/// Hotelling deflation.
///
/// Reuses the same power-iteration scheme as [`sphere_dominant_axis`] (so no new
/// linear-algebra dependency is introduced): repeatedly extract the current
/// dominant eigenvector, then deflate it out of the matrix (`M ← M − λ a aᵀ`) so
/// the next pass yields the next eigenvector. The returned set spans the full
/// `d`-dimensional eigenbasis, covering the least-dominant (orthogonal) axes that
/// the dominant-only seed never reaches.
fn sphere_eigenbasis(moment: ArrayView2<'_, f64>) -> Vec<Array1<f64>> {
    let d = moment.nrows();
    let mut basis: Vec<Array1<f64>> = Vec::new();
    if d == 0 {
        return basis;
    }
    let mut residual = moment.to_owned();
    for _ in 0..d {
        let axis = match sphere_dominant_axis(residual.view()) {
            Some(a) => a,
            None => break,
        };
        // Rayleigh quotient λ = aᵀ M a for the deflation magnitude.
        let mut ma = Array1::<f64>::zeros(d);
        for r in 0..d {
            let mut acc = 0.0;
            for c in 0..d {
                acc += residual[[r, c]] * axis[c];
            }
            ma[r] = acc;
        }
        let lambda = dot(axis.view(), ma.view());
        if lambda <= 1.0e-12 {
            // Remaining spectrum is (numerically) zero: power iteration can no
            // longer distinguish a direction, so it would just repeat the same
            // axis. Complete the eigenbasis deterministically by Gram–Schmidt over
            // the coordinate axes against the directions already found — this is
            // exactly the null space of `M` (e.g. the pole for an equatorial
            // great-circle spread), which is where the true Fréchet mean lives.
            for k in 0..d {
                let mut cand = Array1::<f64>::zeros(d);
                cand[k] = 1.0;
                for b in &basis {
                    let proj = dot(cand.view(), b.view());
                    for col in 0..d {
                        cand[col] -= proj * b[col];
                    }
                }
                let nrm = norm(cand.view());
                if nrm > 1.0e-9 {
                    let unit = cand.mapv(|x| x / nrm);
                    basis.push(unit);
                }
            }
            break;
        }
        basis.push(axis.clone());
        for r in 0..d {
            for c in 0..d {
                residual[[r, c]] -= lambda * axis[r] * axis[c];
            }
        }
    }
    basis
}

/// Build the weighted second-moment matrix `M = Σ wᵢ pᵢ pᵢᵀ = Pᵀ diag(w) P`.
///
/// This is a single weighted cross-product over ALL `n` points, so it routes
/// through [`gam_linalg::faer_ndarray::fast_xt_diag_x`], whose auto-dispatch
/// shim runs the `Pᵀ diag(w) P` Gram on the GPU (`crate::gpu::try_fast_xt_diag_x`)
/// when the batch is large enough and otherwise on faer. The result is bit-for-bit
/// the same `d×d` symmetric Gram as the explicit triple loop (f64 throughout).
fn sphere_second_moment(values: ArrayView2<'_, f64>, weights: ArrayView1<'_, f64>) -> Array2<f64> {
    gam_linalg::faer_ndarray::fast_xt_diag_x(&values, &weights)
}

/// Dominant eigenvector of a symmetric PSD matrix via power iteration.
fn sphere_dominant_axis(moment: ArrayView2<'_, f64>) -> Option<Array1<f64>> {
    let d = moment.nrows();
    if d == 0 {
        return None;
    }
    let mut v = Array1::<f64>::from_elem(d, 1.0 / (d as f64).sqrt());
    for _ in 0..SPHERE_DOMINANT_AXIS_POWER_ITERS {
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
            return None;
        }
        nv.mapv_inplace(|x| x / nrm);
        v = nv;
    }
    let nrm = norm(v.view());
    if nrm > 0.0 {
        Some(v.mapv(|x| x / nrm))
    } else {
        None
    }
}

/// Deterministic equatorial minimizer for a non-identifiable (antipodal /
/// degenerate) Fréchet problem.
///
/// When the data's second-moment matrix has its mass concentrated along a single
/// axis `a` (e.g. equal-weight `{e1, −e1}` gives `M = diag(1,0,…)`), the Fréchet
/// objective `½ Σ wᵢ d(μ, pᵢ)²` is minimized by the ENTIRE great subsphere
/// orthogonal to `a` — every point of that equator is an exact minimizer, so no
/// log-map iteration can converge to a unique point. Rather than reporting the
/// problem as unsolvable (which would contradict the documented contract), pick
/// one minimizer that is fully determined by the inputs:
///
///   1. `a` = dominant eigenvector of `M = Σ wᵢ pᵢ pᵢᵀ` (the antipodal axis).
///   2. Among the coordinate axes `e_k`, pick the one LEAST aligned with the
///      data, i.e. the smallest diagonal moment `M[k,k]`, tie-broken by the
///      lowest coordinate index `k`.
///   3. Project `e_k` onto the orthogonal complement of `a` and normalize; the
///      result lies on the equator (hence is a true minimizer) and is uniquely
///      determined by the inputs.
///
/// Returns `None` only when no equatorial direction can be formed (degenerate
/// dimension), in which case the caller surfaces the genuine error.
fn sphere_equatorial_minimizer(
    values: ArrayView2<'_, f64>,
    weights: ArrayView1<'_, f64>,
) -> Option<Array1<f64>> {
    let (_, d) = values.dim();
    if d == 0 {
        return None;
    }
    let moment = sphere_second_moment(values, weights);
    let axis = sphere_dominant_axis(moment.view())?;
    // Choose the coordinate axis least aligned with the data (smallest diagonal
    // second moment), tie-broken by lowest index.
    let mut best_k = 0usize;
    let mut best_diag = moment[[0, 0]];
    for k in 1..d {
        let diag = moment[[k, k]];
        if diag < best_diag {
            best_diag = diag;
            best_k = k;
        }
    }
    // Project e_{best_k} onto the orthogonal complement of `axis`, then onto the
    // complements of any further degenerate directions by simply normalizing the
    // residual; for a rank-1 concentration this single projection suffices.
    let mut cand = Array1::<f64>::zeros(d);
    cand[best_k] = 1.0;
    let proj = dot(cand.view(), axis.view());
    for col in 0..d {
        cand[col] -= proj * axis[col];
    }
    let nrm = norm(cand.view());
    if nrm > 0.0 {
        return Some(cand.mapv(|x| x / nrm));
    }
    // `e_{best_k}` was parallel to `axis`; fall back to the first coordinate axis
    // whose residual after projection is non-degenerate (lowest index wins).
    for k in 0..d {
        let mut c = Array1::<f64>::zeros(d);
        c[k] = 1.0;
        let p = dot(c.view(), axis.view());
        for col in 0..d {
            c[col] -= p * axis[col];
        }
        let n = norm(c.view());
        if n > 0.0 {
            return Some(c.mapv(|x| x / n));
        }
    }
    None
}

fn sphere_weighted_log_step(
    values: ArrayView2<'_, f64>,
    weights: ArrayView1<'_, f64>,
    base: ArrayView1<'_, f64>,
) -> Result<Array1<f64>, String> {
    let mut step = Array1::<f64>::zeros(base.len());
    for row in 0..values.nrows() {
        let mut dot_value = 0.0_f64;
        let mut chord_sq = 0.0_f64;
        for col in 0..base.len() {
            dot_value += values[[row, col]] * base[col];
            let d = values[[row, col]] - base[col];
            chord_sq += d * d;
        }
        let dot_value = dot_value.clamp(-1.0, 1.0);
        if dot_value <= -1.0 + 1.0e-12 {
            return Err("spherical log map is undefined at antipodal points".to_string());
        }
        // Chord form theta = 2·arcsin(|v-base|/2) avoids the acos(p·q)
        // cancellation for nearby points (see SphereManifold::log_map); the
        // dot product is still used for the tangent projection below.
        let theta = 2.0 * (0.5 * chord_sq.sqrt()).min(1.0).asin();
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
        // Chord form theta = 2·arcsin(|v-base|/2) avoids the acos(p·q)
        // cancellation for nearby points, so the Fréchet objective stays
        // accurate as the mean iteration converges (rows collapse onto base).
        let mut chord_sq = 0.0_f64;
        for col in 0..base.len() {
            let d = values[[row, col]] - base[col];
            chord_sq += d * d;
        }
        let theta = 2.0 * (0.5 * chord_sq.sqrt()).min(1.0).asin();
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
    if let Some(mu) = best_mu {
        return Ok(mu.to_vec());
    }
    // No log-map iteration converged: the problem is non-identifiable because the
    // data has a degenerate/antipodal structure (e.g. equal-weight {e1, −e1},
    // whose minimizer set is the entire orthogonal equator). Honor the documented
    // contract by returning ONE deterministic equatorial minimizer rather than an
    // endpoint surrogate or a "not identifiable" error.
    if let Some(mu) = sphere_equatorial_minimizer(y.view(), w.view()) {
        return Ok(mu.to_vec());
    }
    // Truly no minimizer can be formed (degenerate dimension); surface the error.
    Err("spherical Fréchet mean is not identifiable for these points".to_string())
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::array;

    fn obj_at(values: ArrayView2<'_, f64>, weights: ArrayView1<'_, f64>, mu: &[f64]) -> f64 {
        let mu_arr = Array1::from(mu.to_vec());
        sphere_frechet_objective(values, weights, mu_arr.view())
    }

    #[test]
    fn antipodal_pair_returns_deterministic_equatorial_minimizer() {
        // Equal-weight {e1, -e1} on S^2: the Fréchet objective is minimized by the
        // ENTIRE equator orthogonal to e1, so no log-map iteration converges. The
        // tie-breaker must return one deterministic minimizer on that equator
        // rather than the "not identifiable" error.
        let values = array![[1.0, 0.0, 0.0], [-1.0, 0.0, 0.0]];
        let mean = sphere_frechet_mean(values.view(), None, 1.0e-12, 256)
            .expect("antipodal pair must return a deterministic minimizer");
        assert_eq!(mean.len(), 3);

        // It is a unit vector.
        let nrm = (mean[0] * mean[0] + mean[1] * mean[1] + mean[2] * mean[2]).sqrt();
        assert!(
            (nrm - 1.0).abs() < 1e-9,
            "mean must be a unit vector, got {nrm}"
        );

        // It lies on the equator orthogonal to the antipodal axis e1.
        assert!(
            mean[0].abs() < 1e-9,
            "mean must be orthogonal to e1, got {mean:?}"
        );

        // The dominant data axis is e1 (col 0); the least-aligned coordinate axis
        // is e2 (col 1, lowest index among the zero-moment axes). The projection of
        // e2 onto the complement of e1 is e2 itself, so the deterministic pick is
        // exactly +e2.
        assert!((mean[1] - 1.0).abs() < 1e-9, "expected +e2, got {mean:?}");
        assert!(mean[2].abs() < 1e-9, "expected +e2, got {mean:?}");

        // And it is genuinely a minimizer: its objective ties the equatorial value
        // pi^2/2 attained by e.g. e2 and by e3, and is strictly below the value at
        // an endpoint e1 (which is NOT a minimizer for this data).
        let w = normalize_weights(2, None).unwrap();
        let y = normalize_sphere_matrix(values.view()).unwrap();
        let obj_mean = obj_at(y.view(), w.view(), &mean);
        let obj_e3 = obj_at(y.view(), w.view(), &[0.0, 0.0, 1.0]);
        let obj_e1 = obj_at(y.view(), w.view(), &[1.0, 0.0, 0.0]);
        assert!(
            (obj_mean - obj_e3).abs() < 1e-9,
            "equatorial minimizer must tie other equatorial points: {obj_mean} vs {obj_e3}"
        );
        assert!(
            obj_mean < obj_e1 - 1e-9,
            "equatorial minimizer must beat an endpoint: {obj_mean} vs {obj_e1}"
        );
    }

    #[test]
    fn antipodal_minimizer_is_deterministic_across_calls() {
        let values = array![[1.0, 0.0, 0.0], [-1.0, 0.0, 0.0]];
        let a = sphere_frechet_mean(values.view(), None, 1.0e-12, 256).unwrap();
        let b = sphere_frechet_mean(values.view(), None, 1.0e-12, 256).unwrap();
        assert_eq!(a, b, "tie-breaker must be deterministic across calls");
    }

    #[test]
    fn empty_input_still_errors() {
        // Zero-weight / empty input has no minimizer; the genuine error must remain.
        let values = array![[1.0, 0.0, 0.0]];
        let zero = array![0.0_f64];
        let err = sphere_frechet_mean(values.view(), Some(zero.view()), 1.0e-12, 256);
        assert!(err.is_err(), "zero-weight input must still error");
    }

    #[test]
    fn non_degenerate_mean_unchanged() {
        // A clearly identifiable cluster must still converge to the ordinary
        // Karcher mean, not the equatorial fallback.
        let values = array![[1.0, 0.0, 0.0], [0.9, 0.1, 0.0], [0.9, 0.0, 0.1]];
        let mean = sphere_frechet_mean(values.view(), None, 1.0e-12, 256).unwrap();
        // Mean should be close to e1 (dominant direction), not on the equator.
        assert!(mean[0] > 0.9, "expected near-e1 mean, got {mean:?}");
    }

    #[test]
    fn sectional_curvature_is_one_on_nondegenerate_plane() {
        // S^2 has constant sectional curvature +1 on any genuine tangent plane.
        let m = SphereManifold::new(2);
        let point = array![1.0, 0.0, 0.0];
        // Two orthogonal tangent vectors at e1.
        let u = array![0.0, 1.0, 0.0];
        let v = array![0.0, 0.0, 1.0];
        let k = m
            .sectional_curvature(point.view(), (u.view(), v.view()))
            .expect("unit sphere has defined curvature on a nondegenerate plane");
        assert!((k - 1.0).abs() < 1.0e-12, "expected +1, got {k}");
    }

    #[test]
    fn sectional_curvature_is_singular_for_collinear_pair() {
        let m = SphereManifold::new(2);
        let point = array![1.0, 0.0, 0.0];
        let u = array![0.0, 1.0, 0.0];
        // v parallel to u in the tangent space: zero parallelogram area.
        let v = array![0.0, 2.0, 0.0];
        match m.sectional_curvature(point.view(), (u.view(), v.view())) {
            Err(GeometryError::Singular(_)) => {}
            other => panic!("expected Singular for collinear pair, got {other:?}"),
        }
    }

    #[test]
    fn sectional_curvature_is_singular_for_purely_radial_pair() {
        // Vectors that vanish after projecting off the radial direction span no
        // tangent plane, even though they look independent in ambient space.
        let m = SphereManifold::new(2);
        let point = array![1.0, 0.0, 0.0];
        let u = array![1.0, 0.0, 0.0];
        let v = array![2.0, 0.0, 0.0];
        match m.sectional_curvature(point.view(), (u.view(), v.view())) {
            Err(GeometryError::Singular(_)) => {}
            other => panic!("expected Singular for radial pair, got {other:?}"),
        }
    }

    #[test]
    fn sectional_curvature_is_unsupported_below_two_dimensions() {
        // S^1 has a one-dimensional tangent space — no 2-plane exists.
        let m = SphereManifold::new(1);
        let point = array![1.0, 0.0];
        let u = array![0.0, 1.0];
        let v = array![0.0, 1.0];
        match m.sectional_curvature(point.view(), (u.view(), v.view())) {
            Err(GeometryError::Unsupported(_)) => {}
            other => panic!("expected Unsupported on S^1, got {other:?}"),
        }
    }

    #[test]
    fn sectional_curvature_rejects_non_unit_base_point() {
        let m = SphereManifold::new(2);
        let point = array![2.0, 0.0, 0.0];
        let u = array![0.0, 1.0, 0.0];
        let v = array![0.0, 0.0, 1.0];
        match m.sectional_curvature(point.view(), (u.view(), v.view())) {
            Err(GeometryError::InvalidPoint(_)) => {}
            other => panic!("expected InvalidPoint on non-unit base, got {other:?}"),
        }
    }
}
