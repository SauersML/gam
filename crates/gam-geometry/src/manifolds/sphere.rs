use faer::Side;
use gam_linalg::faer_ndarray::FaerEigh;
use ndarray::{Array1, Array2, ArrayView1, ArrayView2};

use crate::manifold::{
    GeometryError, GeometryResult, RiemannianManifold, check_len, dot, identity, norm,
};
use crate::normalize_weights;

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SphereManifold {
    intrinsic_dim: usize,
}

impl SphereManifold {
    pub const fn new(intrinsic_dim: usize) -> Self {
        Self { intrinsic_dim }
    }

    /// Reject points that are not on the unit sphere. This guards the maps
    /// that are only meaningful at a genuine manifold point — `log_map`,
    /// `metric_tensor`, `parallel_transport`, `tangent_basis`, `project_tangent`
    /// — where a non-unit `p` makes `v − p(pᵀv)` not even tangent. It is
    /// deliberately *not* applied to `exp_map` / `exp_map_vjp`, the forward and
    /// adjoint pair used by reverse-mode autodiff: they accept any nonzero
    /// ambient `point`, so finite-difference probes can step off the sphere
    /// (see `exp_map`).
    ///
    /// The band is the widest `|‖p‖² − 1|` an f64 normalization leaves
    /// ([`unit_normalization_band`](gam_math::roundoff::unit_normalization_band)).
    /// Iterates meet it by construction, because `exp_map` normalizes its
    /// output, so a larger defect names an input that was not normalized in
    /// f64. It is refused with the measured defect, never normalized silently.
    fn require_unit(&self, point: ArrayView1<'_, f64>) -> GeometryResult<()> {
        let n2 = dot(point, point);
        let squared_norm_defect = (n2 - 1.0).abs();
        let band = gam_math::roundoff::unit_normalization_band(point.len());
        if !(squared_norm_defect <= band) {
            return Err(GeometryError::PointOffUnitSphere {
                context: "sphere operation",
                squared_norm_defect,
                band,
            });
        }
        Ok(())
    }

    /// The geodesic `cos θ·p + (sin θ/θ)·ξ` in ambient coordinates, with
    /// `ξ = v − (p·v)p` and `θ = ‖ξ‖`, before `exp_map` normalizes it.
    fn ambient_geodesic(point: ArrayView1<'_, f64>, tangent_vec: ArrayView1<'_, f64>) -> Array1<f64> {
        let c = dot(point, tangent_vec);
        let xi = &tangent_vec.to_owned() - &(point.to_owned() * c);
        let theta = norm(xi.view());
        if Self::geodesic_is_small_angle(theta) {
            // Numerically-stable evaluation of the geodesic below as θ→0: once
            // θ² ≤ ε, cos(θ) and sin(θ)/θ both round to within u of 1 (see
            // `geodesic_is_small_angle`), so the map degenerates to `p + ξ`. This keeps
            // the geodesic a SINGLE map, bit-identical and C¹ across the branch
            // boundary, with a matching adjoint in `exp_map_vjp`.
            return &point + &xi;
        }
        point.to_owned() * theta.cos() + xi * (theta.sin() / theta)
    }

    /// Whether the geodesic `cos θ·p + (sin θ/θ)·ξ` equals its small-angle form
    /// `p + ξ` to rounding. The corrections are `θ²/2` in `cos θ` and `θ²/6` in
    /// `sin θ/θ`, so both round to within `u = ε/2` of 1 once `θ² ≤ ε`.
    /// `exp_map` and `exp_map_vjp` branch on this one predicate, so the value and
    /// its adjoint switch at the same θ.
    fn geodesic_is_small_angle(theta: f64) -> bool {
        theta * theta <= f64::EPSILON
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
        // `u` differs from `sign·p` only in its anchor entry, one subtraction of
        // magnitude at most `|p_anchor| + 1`; with its `m`-term norm it rounds by
        // at most `γ_{m+1}·(|p_anchor| + 1)`, so a smaller `u` is `p` on its axis.
        if u_nrm <= gam_linalg::roundoff::accumulation_growth(m + 1) * (max_abs + 1.0) {
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
        // `point` need not satisfy ‖p‖ = 1: this is the forward whose reverse
        // mode [`exp_map_vjp`](Self::exp_map_vjp) differentiates, and the
        // finite-difference pins (`tests/sphere_exp_map_vjp_matches_finite_difference.rs`)
        // step `point` off the unit sphere on purpose, so the map must stay
        // smooth in the raw ambient coordinates rather than reject them. With
        // `c = p·v` the tangent component is `ξ = v − c·p` (the orthogonal
        // projection only when ‖p‖ = 1).
        //
        // The ambient geodesic is then normalized. At a unit `p` it already has
        // norm 1 in exact arithmetic, so normalizing changes nothing but the
        // rounding, and it keeps every output inside the f64 normalization band
        // that `require_unit` enforces: iterates of any length cannot drift off
        // the sphere one rounding per step.
        let geodesic = Self::ambient_geodesic(point, tangent_vec);
        let length = norm(geodesic.view());
        if !(length > 0.0 && length.is_finite()) {
            return Err(GeometryError::Singular(
                "sphere exponential of a point whose ambient geodesic has no direction",
            ));
        }
        Ok(geodesic / length)
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
        if theta == 0.0 {
            return Ok(Array1::<f64>::zeros(m));
        }
        let mut u = &p_to - &(p_from.to_owned() * c);
        let u_nrm = norm(u.view());
        // `u = p_to − (p_from·p_to)·p_from` is an m-term inner product and one
        // subtraction per coordinate, so its rounding is at most
        // `γ_{m+2}·(‖p_to‖₁ + |c|·‖p_from‖₁)` in ℓ1, which bounds it in ℓ2 too.
        let p_from_l1: f64 = p_from.iter().map(|v| v.abs()).sum();
        let p_to_l1: f64 = p_to.iter().map(|v| v.abs()).sum();
        let resolution = gam_linalg::roundoff::accumulation_growth(m + 2)
            * (p_to_l1 + c.abs() * p_from_l1);
        if !(u_nrm > resolution) {
            if c > 0.0 {
                // p_to coincides with p_from to rounding, so the logarithm is
                // zero within that band.
                return Ok(Array1::<f64>::zeros(m));
            }
            // An unresolvable tangent direction opposite p_from means p_to is
            // its antipode. The logarithm there is multivalued — every unit
            // u ⟂ p_from satisfies Exp_{p_from}(πu) = −p_from — so there is no
            // single correct answer to return. Surface it rather than
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
        // `1 + from·to` accumulates an m-term inner product and one addition, so
        // it rounds within `γ_{m+1}·(1 + Σ|from_i·to_i|)`. A denominator inside
        // that band leaves the endpoints antipodal to rounding.
        let mut absolute_terms = 1.0_f64;
        for i in 0..m {
            absolute_terms += (from[i] * to[i]).abs();
        }
        if !(denom.abs() > gam_linalg::roundoff::accumulation_growth(m + 1) * absolute_terms) {
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

    /// Induced metric: `G·v = v` at a unit base point.
    fn metric_product(
        &self,
        point: ArrayView1<'_, f64>,
        tangent: ArrayView1<'_, f64>,
    ) -> GeometryResult<Array1<f64>> {
        check_len("Sphere metric point", point.len(), self.ambient_dim())?;
        check_len("Sphere metric tangent", tangent.len(), self.ambient_dim())?;
        self.require_unit(point)?;
        Ok(tangent.to_owned())
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
        // The area cancels exactly on a collinear pair. After the radial
        // projections and the `m`-term inner products, two products and one
        // subtraction, it rounds by at most `γ_{4m+3}·(uu·vv + uv²)`; an area
        // inside that band spans no tangent plane.
        let area_band = gam_linalg::roundoff::accumulation_growth(4 * point.len() + 3)
            * (uu * vv + uv * uv);
        if !area_sq.is_finite() || area_sq <= area_band {
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
    /// [`Self::project_tangent`]. (The metric-raising default would give the same
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
        // No `require_unit` here: this is the exact adjoint of
        // [`exp_map`](Self::exp_map), which accepts any nonzero ambient `point`.
        // The general branch below carries the `c(1 − |p|²)` terms, so it is
        // correct for any ambient `p`; gating on ‖p‖ = 1 would make that branch
        // unreachable and break the off-sphere finite-difference contract the VJP
        // exists for.
        //
        // `exp_map` returns `y = ỹ/‖ỹ‖` with `ỹ` the ambient geodesic. The
        // normalization pulls a cotangent back to `(g − y(y·g))/‖ỹ‖` on `ỹ`, and
        // the closed form below pulls that back through the geodesic.
        let geodesic = Self::ambient_geodesic(point, tangent_vec);
        let length = norm(geodesic.view());
        if !(length > 0.0 && length.is_finite()) {
            return Err(GeometryError::Singular(
                "sphere exponential of a point whose ambient geodesic has no direction",
            ));
        }
        let unit = &geodesic / length;
        let radial = dot(unit.view(), grad_output);
        let geodesic_cotangent = (&grad_output.to_owned() - &(&unit * radial)) / length;

        // Forward geodesic: with `xi = (I - p p^T) v`, `theta = |xi|`,
        //   ỹ = cos(theta) p + (sin(theta)/theta) xi.
        // We differentiate this closed form and return the transpose-applied
        // (vector–Jacobian) products w.r.t. the base point `p` and the raw
        // (unprojected) tangent input `v`.
        let c = dot(point, tangent_vec); // p · v
        let xi = &tangent_vec.to_owned() - &(point.to_owned() * c);
        let theta = norm(xi.view());
        let g = geodesic_cotangent.view();
        let p = point;
        let v = tangent_vec;

        // Small-theta branch of the FORWARD is the geodesic limit `y = p + xi`
        // (see `exp_map`), with `xi = v - c p`, `c = p·v`. Its exact Jacobians
        // are
        //   dy/dv = I - p p^T,     dy/dp = (1 - c) I - p v^T,
        // so the transpose-applied pullbacks of cotangent `g` are
        //   grad_v = (I - p p^T) g = g - p (p·g),
        //   grad_p = (1 - c) g - v (p·g).
        // These are exactly the theta -> 0 limit of the general branch below, so
        // the VJP is continuous at the switch. At p = g = e1, xi = 0 they give
        // grad_v = (I - p p^T) g = 0 — the correct radial derivative. The switch
        // is `exp_map`'s own, `geodesic_is_small_angle`.
        if Self::geodesic_is_small_angle(theta) {
            let p_dot_g = dot(p, g);
            let grad_v = &g.to_owned() - &(p.to_owned() * p_dot_g);
            let grad_p = &(&g.to_owned() * (1.0 - c)) - &(v.to_owned() * p_dot_g);
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

pub(crate) fn validate_sphere_matrix(values: ArrayView2<'_, f64>) -> Result<(), String> {
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

/// The rows of `values` as points on the unit sphere. Each row must already be
/// unit-norm within the band an f64 normalization leaves
/// ([`unit_normalization_band`](gam_math::roundoff::unit_normalization_band)).
/// That is the entry rule `SphereManifold` applies to its points, so `"sphere"`
/// and `stiefel(k=1)` accept the same data. A row off the sphere is refused
/// with its measured defect, never normalized silently.
pub(crate) fn require_unit_sphere_rows(values: ArrayView2<'_, f64>) -> Result<Array2<f64>, String> {
    validate_sphere_matrix(values)?;
    let band = gam_math::roundoff::unit_normalization_band(values.ncols());
    for (row, point) in values.outer_iter().enumerate() {
        let squared_norm_defect = (dot(point, point) - 1.0).abs();
        if !(squared_norm_defect <= band) {
            return Err(format!(
                "spherical row {row} is not unit-norm in f64: |‖y‖² − 1| = \
                 {squared_norm_defect:.3e} exceeds {band:.3e}, the widest an f64 normalization \
                 leaves it; normalize each row in f64 before passing it, e.g. y / ‖y‖"
            ));
        }
    }
    Ok(values.to_owned())
}

/// Batched Riemannian log map of each row of `values` at a single `base`, in
/// ambient tangent coordinates. Rows and base must be unit-norm under the same
/// entry rule as the [`SphereManifold::log_map`] trait method
/// (`require_unit_sphere_rows`); the geodesic angle uses the numerically stable
/// `atan2(|u|, p·q)` form. Errors at antipodal points. This is the
/// response-geometry companion to the trait method.
pub fn response_sphere_log_map(
    values: ArrayView2<'_, f64>,
    base: ArrayView1<'_, f64>,
) -> Result<Array2<f64>, String> {
    let y = require_unit_sphere_rows(values)?;
    let base2 = Array2::from_shape_fn((1, base.len()), |(_, j)| base[j]);
    let b_mat = require_unit_sphere_rows(base2.view())?;
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
    let base_l1: f64 = b_mat.row(0).iter().map(|v| v.abs()).sum();
    // `u = q − (p·q)·p` is a d-term inner product and one subtraction per
    // coordinate, so its rounding is at most `γ_{d+2}·(‖q‖₁ + |p·q|·‖p‖₁)` in
    // ℓ1, which bounds it in ℓ2 too.
    let growth = gam_linalg::roundoff::accumulation_growth(d + 2);
    for row in 0..n {
        let mut dot = dots[row];
        dot = dot.clamp(-1.0, 1.0);
        // Geodesic angle via theta = atan2(|u|, p·q) with u = q − (p·q)p, the
        // component of q orthogonal to p (|u| = sin theta). For nearby points
        // p·q rounds to exactly 1.0 in f64 and acos(p·q) collapses a genuine
        // ~1e-9 distance to 0; |u| is formed straight from the coordinates with
        // no near-1 subtraction, so atan2(|u|, p·q) ≈ |u| stays accurate and
        // the tangent norm equals the geodesic distance as documented.
        let mut s_sq = 0.0_f64;
        let mut row_l1 = 0.0_f64;
        for col in 0..d {
            let uc = y[[row, col]] - dot * b_mat[[0, col]];
            s_sq += uc * uc;
            row_l1 += y[[row, col]].abs();
        }
        let s = s_sq.sqrt();
        if !(s > growth * (row_l1 + dot.abs() * base_l1)) {
            // The tangent direction is inside its own rounding band. Opposite
            // `base` that is the antipode, where the log map has no direction;
            // beside `base` the row coincides with it and its log is zero.
            if dot < 0.0 {
                return Err("spherical log map is undefined at antipodal points".to_string());
            }
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
/// points on the unit sphere. The base must be unit-norm
/// (`require_unit_sphere_rows`); the orthogonal
/// component of the tangent drives the geodesic step `cos(r)·p + (sin r / r)·z`.
/// This is the response-geometry companion to [`SphereManifold::exp_map`].
pub fn response_sphere_exp_map(
    tangent: ArrayView2<'_, f64>,
    base: ArrayView1<'_, f64>,
) -> Result<Array2<f64>, String> {
    let base2 = Array2::from_shape_fn((1, base.len()), |(_, j)| base[j]);
    let b_mat = require_unit_sphere_rows(base2.view())?;
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
        if r == 0.0 {
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

fn sphere_mean_candidates(
    values: ArrayView2<'_, f64>,
    weights: ArrayView1<'_, f64>,
) -> Result<Vec<Array1<f64>>, String> {
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
    if let Some(unit) = sphere_dominant_axis(moment.view()) {
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

/// Orthonormal eigenbasis of a symmetric PSD matrix, from one symmetric
/// eigendecomposition.
///
/// The returned set spans the full `d`-dimensional space, covering the
/// least-dominant (orthogonal) axes that the dominant-only seed never reaches,
/// including the null space of `M` (the pole for an equatorial great-circle
/// spread), which is where the true Fréchet mean lives. A decomposition that
/// fails contributes no seeds; the extrinsic and dominant-axis seeds remain.
fn sphere_eigenbasis(moment: ArrayView2<'_, f64>) -> Vec<Array1<f64>> {
    sphere_moment_eigensystem(moment)
        .map(|system| {
            system
                .1
                .columns()
                .into_iter()
                .map(|column| column.to_owned())
                .collect()
        })
        .unwrap_or_default()
}

/// The symmetric eigendecomposition `(values, vectors)` of a moment matrix, or
/// `None` for a decomposition that fails.
fn sphere_moment_eigensystem(moment: ArrayView2<'_, f64>) -> Option<(Array1<f64>, Array2<f64>)> {
    moment.eigh(Side::Lower).ok()
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

/// Dominant eigenvector of a symmetric PSD matrix: the eigenvector of its
/// largest eigenvalue from one symmetric eigendecomposition, or `None` when that
/// eigenvalue is not positive (a zero matrix has no dominant direction).
fn sphere_dominant_axis(moment: ArrayView2<'_, f64>) -> Option<Array1<f64>> {
    let (values, vectors) = sphere_moment_eigensystem(moment)?;
    if values.is_empty() {
        return None;
    }
    let mut top = 0usize;
    for index in 1..values.len() {
        if values[index] > values[top] {
            top = index;
        }
    }
    if !(values[top] > 0.0) {
        return None;
    }
    Some(vectors.column(top).to_owned())
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

/// The weighted spherical log step `Σ w·θ·u/‖u‖` at `base`, together with its
/// rounding band: an ℓ1 bound, hence an ℓ2 bound, on how far the computed step
/// can sit from the exact one.
///
/// A contributing row adds its direction's rounding `resolution/‖u‖` scaled by
/// `w·θ`, plus `γ_{n+2d+8}·√d·w·θ` for forming `θ`, `‖u‖`, the scale and the row
/// sum (`‖û‖₁ ≤ √d`). A row skipped because it coincides with `base` adds `w·θ`,
/// which bounds the contribution it leaves out.
fn sphere_weighted_log_step(
    values: ArrayView2<'_, f64>,
    weights: ArrayView1<'_, f64>,
    base: ArrayView1<'_, f64>,
) -> Result<(Array1<f64>, f64), String> {
    let d = base.len();
    let n = values.nrows();
    let base_l1: f64 = base.iter().map(|b| b.abs()).sum();
    // The tangent component `v − (v·base)·base` is formed by a d-term inner
    // product and one subtraction per coordinate, so its rounding is at most
    // `γ_{d+2}·(‖v‖₁ + |v·base|·‖base‖₁)` in ℓ1, which bounds it in ℓ2 too.
    let growth = gam_linalg::roundoff::accumulation_growth(d + 2);
    let formation =
        gam_linalg::roundoff::accumulation_growth(n + 2 * d + 8) * (d as f64).sqrt();
    let mut step = Array1::<f64>::zeros(d);
    let mut band = 0.0_f64;
    let mut tangent = Array1::<f64>::zeros(d);
    for row in 0..n {
        let mut dot_value = 0.0_f64;
        let mut chord_sq = 0.0_f64;
        let mut row_l1 = 0.0_f64;
        for col in 0..d {
            dot_value += values[[row, col]] * base[col];
            let diff = values[[row, col]] - base[col];
            chord_sq += diff * diff;
            row_l1 += values[[row, col]].abs();
        }
        let dot_value = dot_value.clamp(-1.0, 1.0);
        // Chord form theta = 2·arcsin(|v-base|/2) avoids the acos(p·q)
        // cancellation for nearby points (see SphereManifold::log_map).
        let theta = 2.0 * (0.5 * chord_sq.sqrt()).min(1.0).asin();
        if theta == 0.0 {
            continue;
        }
        for col in 0..d {
            tangent[col] = values[[row, col]] - dot_value * base[col];
        }
        let tangent_norm = norm(tangent.view());
        let resolution = growth * (row_l1 + dot_value.abs() * base_l1);
        if !(tangent_norm > resolution) {
            // The tangent direction is inside its own rounding band. Opposite
            // `base` that is the antipode, where the log map has no direction;
            // beside `base` the row coincides with it and its log is zero within
            // that band.
            if dot_value < 0.0 {
                return Err("spherical log map is undefined at antipodal points".to_string());
            }
            band += weights[row].abs() * theta;
            continue;
        }
        // `θ·û` with `û = u/‖u‖`: the stable magnitude `‖u‖` stands in for `sin θ`.
        let scale = theta / tangent_norm;
        for col in 0..d {
            step[col] += weights[row] * tangent[col] * scale;
        }
        band += weights[row].abs() * theta * (resolution / tangent_norm + formation);
    }
    Ok((step, band))
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
    if r == 0.0 {
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
) -> Result<Vec<f64>, String> {
    let y = require_unit_sphere_rows(points)?;
    let w = normalize_weights(y.nrows(), weights)?;
    let mut candidates = sphere_mean_candidates(y.view(), w.view())?;
    if candidates.is_empty() {
        candidates.push(sphere_orthogonal_unit(y.row(0))?);
    }
    let mut best_mu: Option<Array1<f64>> = None;
    let mut best_obj = f64::INFINITY;
    let mut antipodal_refusal = false;
    for candidate in candidates {
        let mut mu = candidate;
        // From each seed, step until the step is inside its own rounding band,
        // which certifies that `mu` is stationary. A step that stops shrinking
        // above that band means rounding ended the iteration before
        // stationarity, and the seed is refused.
        let mut previous_norm = f64::INFINITY;
        let certified = loop {
            let (step, band) = match sphere_weighted_log_step(y.view(), w.view(), mu.view()) {
                Ok(pair) => pair,
                Err(_) => {
                    antipodal_refusal = true;
                    break false;
                }
            };
            let step_norm = norm(step.view());
            if step_norm <= band {
                break true;
            }
            if !(step_norm < previous_norm) {
                break false;
            }
            previous_norm = step_norm;
            mu = sphere_exp_single(step.view(), mu.view())?;
        };
        if !certified {
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
    if !antipodal_refusal {
        return Err(
            "spherical Fréchet mean: no seed reached a step inside its own rounding band"
                .to_string(),
        );
    }
    // Every certified route was blocked by an antipode: the problem is
    // non-identifiable because the data has a degenerate/antipodal structure
    // (e.g. equal-weight {e1, −e1}, whose minimizer set is the entire
    // orthogonal equator). Honor the documented contract by returning ONE
    // deterministic equatorial minimizer rather than an endpoint surrogate or a
    // "not identifiable" error.
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
        let mean = sphere_frechet_mean(values.view(), None)
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
        let y = require_unit_sphere_rows(values.view()).unwrap();
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
        let a = sphere_frechet_mean(values.view(), None).unwrap();
        let b = sphere_frechet_mean(values.view(), None).unwrap();
        assert_eq!(a, b, "tie-breaker must be deterministic across calls");
    }

    #[test]
    fn empty_input_still_errors() {
        // Zero-weight / empty input has no minimizer; the genuine error must remain.
        let values = array![[1.0, 0.0, 0.0]];
        let zero = array![0.0_f64];
        let err = sphere_frechet_mean(values.view(), Some(zero.view()));
        assert!(err.is_err(), "zero-weight input must still error");
    }

    #[test]
    fn non_degenerate_mean_unchanged() {
        // A clearly identifiable cluster must still converge to the ordinary
        // Karcher mean, not the equatorial fallback.
        let mut values = array![[1.0, 0.0, 0.0], [0.9, 0.1, 0.0], [0.9, 0.0, 0.1]];
        for mut row in values.outer_iter_mut() {
            let length = norm(row.view());
            row.mapv_inplace(|value| value / length);
        }
        let mean = sphere_frechet_mean(values.view(), None).unwrap();
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
            Err(GeometryError::PointOffUnitSphere {
                squared_norm_defect,
                band,
                ..
            }) => {
                // ‖(2, 0, 0)‖² − 1 = 3 exactly.
                assert_eq!(squared_norm_defect, 3.0);
                assert_eq!(band, gam_math::roundoff::unit_normalization_band(3));
            }
            other => panic!("expected PointOffUnitSphere on non-unit base, got {other:?}"),
        }
    }

    /// The entry rule is the f64 normalization band. An f64 normalization of an
    /// arbitrary vector passes. A point 1e-5 off the sphere, the size an f32 or
    /// 6-digit normalization leaves, is refused with its measured defect, the
    /// band and the fix. The exponential keeps an iterate inside the band
    /// through 10000 steps.
    #[test]
    fn unit_entry_rule_accepts_f64_normalizations_and_refuses_the_rest() {
        let m = SphereManifold::new(2);
        let raw = array![0.3, -1.7, 2.9];
        let normalized = &raw / norm(raw.view());
        let tangent = array![0.0, 1.0, 0.0];
        assert!(m.log_map(normalized.view(), normalized.view()).is_ok());

        let band = gam_math::roundoff::unit_normalization_band(3);
        // A 1e-5 relative rescale, the size of a 6-digit or f32 normalization.
        let coarse = &normalized * (1.0 + 1.0e-5);
        match m.log_map(normalized.view(), coarse.view()) {
            Err(GeometryError::PointOffUnitSphere {
                squared_norm_defect,
                band: reported,
                ..
            }) => {
                assert!(squared_norm_defect > band);
                assert_eq!(reported, band);
                let message = GeometryError::PointOffUnitSphere {
                    context: "sphere operation",
                    squared_norm_defect,
                    band,
                }
                .to_string();
                assert!(message.contains("p / ‖p‖"), "{message}");
            }
            other => panic!("expected the coarse point to be refused, got {other:?}"),
        }

        let mut point = array![1.0, 0.0, 0.0];
        let step = array![0.0, 0.37, -0.21];
        for _ in 0..10_000 {
            point = m.exp_map(point.view(), step.view()).expect("exp step");
        }
        let squared_norm_defect = (dot(point.view(), point.view()) - 1.0).abs();
        assert!(
            squared_norm_defect <= band,
            "after 10000 exponential steps |‖p‖² − 1| = {squared_norm_defect:.3e} > {band:.3e}"
        );
        assert!(m.log_map(point.view(), point.view()).is_ok());
        assert!(m.project_tangent(point.view(), tangent.view()).is_ok());
    }
}

#[cfg(test)]
mod parallel_transport_tests {
    use super::*;
    use ndarray::array;

    /// Non-antipodal fixture with a genuinely 3-D geodesic (both endpoints
    /// have nonzero components on all three axes, so `parallel_transport`'s
    /// `v − (⟨v,to⟩/(1+⟨from,to⟩))(from+to)` formula is exercised in full,
    /// not on a degenerate in-plane special case). `[`SphereManifold`]`'s
    /// `parallel_transport` had no direct test anywhere in this crate before
    /// this module — `grassmann.rs`'s transport delegates straight to it
    /// (see [`GrassmannManifold::parallel_transport`]) and was likewise
    /// untested.
    fn fixture() -> (SphereManifold, Array1<f64>, Array1<f64>) {
        let m = SphereManifold::new(2);
        let p = array![1.0, 0.0, 0.0];
        let raw = array![1.0, 1.0, 1.0];
        let q = &raw / norm(raw.view());
        (m, p, q)
    }

    fn path2(a: &Array1<f64>, b: &Array1<f64>) -> Array2<f64> {
        let mut out = Array2::<f64>::zeros((2, a.len()));
        out.row_mut(0).assign(a);
        out.row_mut(1).assign(b);
        out
    }

    /// Parallel transport is a linear isometry between tangent spaces:
    /// `⟨Γ(u), Γ(v)⟩ = ⟨u, v⟩`. The sphere carries the embedded (identity)
    /// metric, so the plain ambient dot product is the Riemannian one.
    #[test]
    fn parallel_transport_preserves_inner_product() {
        let (m, p, q) = fixture();
        let path = path2(&p, &q);
        // Tangent at p: orthogonal to p, i.e. zero first coordinate.
        let u = array![0.0, 1.0, 0.4];
        let v = array![0.0, -0.3, 1.2];

        let tu = m.parallel_transport(path.view(), u.view()).expect("Γ(u)");
        let tv = m.parallel_transport(path.view(), v.view()).expect("Γ(v)");

        // Transported vectors must land back in the tangent space at q.
        assert!(
            dot(tu.view(), q.view()).abs() <= 1e-10,
            "Γ(u) not tangent at q"
        );
        assert!(
            dot(tv.view(), q.view()).abs() <= 1e-10,
            "Γ(v) not tangent at q"
        );

        let before = dot(u.view(), v.view());
        let after = dot(tu.view(), tv.view());
        assert!(
            (before - after).abs() <= 1e-10 * before.abs().max(1.0),
            "parallel transport is not an isometry: ⟨u,v⟩={before:.12e}, ⟨Γu,Γv⟩={after:.12e}"
        );
    }

    /// Transporting the initial velocity of the `p→q` geodesic gives the
    /// negative of the `q→p` geodesic's initial velocity:
    /// `Γ_{p→q}(log_p q) = −log_q p`.
    #[test]
    fn parallel_transport_matches_geodesic_velocity_identity() {
        let (m, p, q) = fixture();
        let forward = path2(&p, &q);
        let v_p_to_q = m.log_map(p.view(), q.view()).expect("log_p(q)");
        let v_q_to_p = m.log_map(q.view(), p.view()).expect("log_q(p)");

        let transported = m
            .parallel_transport(forward.view(), v_p_to_q.view())
            .expect("Γ(log_p q)");
        for (i, (&t, &v)) in transported.iter().zip(v_q_to_p.iter()).enumerate() {
            assert!(
                (t + v).abs() <= 1e-9 * v.abs().max(1.0),
                "component {i}: Γ(log_p q)={t:.12e}, −log_q p={:.12e}",
                -v
            );
        }
    }

    /// Transporting forward `p→q` then back `q→p` recovers the original
    /// tangent vector exactly.
    #[test]
    fn parallel_transport_round_trip_is_identity() {
        let (m, p, q) = fixture();
        let forward = path2(&p, &q);
        let backward = path2(&q, &p);
        let u = array![0.0, 0.6, -0.2];

        let out = m
            .parallel_transport(forward.view(), u.view())
            .expect("Γ_{p→q}(u)");
        let back = m
            .parallel_transport(backward.view(), out.view())
            .expect("Γ_{q→p}(Γ_{p→q}(u))");

        for (i, (&b, &orig)) in back.iter().zip(u.iter()).enumerate() {
            assert!(
                (b - orig).abs() <= 1e-9 * orig.abs().max(1.0),
                "component {i}: round-trip {b:.12e} vs original {orig:.12e}"
            );
        }
    }
}
