//! Retraction-based Riemannian optimization for per-point latent coordinates.
//!
//! Each per-row latent coordinate `t_i` lives on a manifold dictated by its
//! basis topology (S¹ for periodic, S² for sphere, torus for tensor of S¹s,
//! interval `[a,b]` for clamped, ℝ for default). Following Absil-Mahony-
//! Sepulchre, we provide:
//!
//!   * a [`Manifold`] trait with smooth retraction `R_p(ξ)`, tangent
//!     projection, vector transport (parallel-transport approximation), and
//!     a Euclidean→Riemannian gradient/Hessian conversion using the
//!     Weingarten map (second fundamental form);
//!   * concrete implementations: [`Euclidean`], [`Circle`], [`Sphere`],
//!     [`Interval`], [`Torus`], [`Product`];
//!   * a serialization-friendly enum [`ManifoldKind`] with [`ManifoldKind::build`].
//!
//! `Euclidean` is the no-op default: all operations are identity / no-op so
//! callers that don't opt into a non-trivial manifold see bit-equivalent
//! behaviour.
//!
//! ## Numerical hardening
//!
//! * Near the south pole of a sphere (small last embedding coordinate),
//!   the canonical chart used by `retract` can amplify error in the tangent
//!   solve. We do not switch charts at runtime; instead we expose a sentinel
//!   threshold ([`SPHERE_POLE_WARN_THRESHOLD`]) and a [`Manifold::warn_at`]
//!   advisory hook for callers that want to log. The typed sentinel
//!   [`ManifoldWarning`] lets callers route warnings to a structured logger.
//! * Near the boundary of an interval, the smooth `tanh` parameterization's
//!   Jacobian blows up. [`Interval`] clips the retraction step to keep the
//!   internal coordinate within a sane band — the clip is applied to the
//!   ambient coordinate *after* the additive update, matching the closed-
//!   constraint retraction in proposal §7.2 (the open `tanh` chart is only
//!   used implicitly through the clip).
//! * [`Manifold::vector_transport`] uses a *parallel-transport approximation*
//!   (re-project the moved tangent vector to the new tangent space). Full
//!   parallel transport along the retraction curve would require an ODE solve;
//!   the projection approximation is a standard Pymanopt-grade tradeoff —
//!   first-order isometric, O(d) per call. Antipodal sphere endpoints
//!   (`<from, to> ≈ -1`) are safe: projection transport degenerates to
//!   "project to T_to M" which is well-defined, but the resulting tangent
//!   can be small; callers that care should check norm-loss after transport.
//!
//! ## Math invariants (per proposal `riemannian_per_point.md`)
//!
//! ```text
//! grad_R = P_p(∇_E)
//! Hess_R[ξ] = P_p(∇²_E · ξ - W_p(ξ, ∇_E))    // ambient (not projected) gradient
//! S¹/S²/S^n   : W_p(ξ, v) = <p, v> ξ
//! Torus T^d   : W_p(ξ, v)_j = <p_j, v_j> ξ_j (block diagonal)
//! Euclidean/Interval : W_p(ξ, v) = 0
//! Product     : W_p blockwise on components
//! ```

use ndarray::{Array1, Array2, ArrayView1, ArrayViewMut1};

const TWO_PI: f64 = std::f64::consts::PI * 2.0;

/// Sentinel: warn when sphere retraction operates this close to a chart
/// singularity (currently only used as an advisory threshold; callers can
/// hook a runtime warning here).
pub const SPHERE_POLE_WARN_THRESHOLD: f64 = 1.0e-8;

/// Typed advisory sentinel returned by [`Manifold::warn_at_typed`] so callers
/// can route to a structured logger without string matching. The simpler
/// [`Manifold::warn_at`] string accessor remains available for ad-hoc logs.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ManifoldWarning {
    /// Point sits close to a chart pole on a sphere; retraction may amplify
    /// numerical error in the canonical chart.
    SphereNearPole,
    /// Point sits within the edge band of an interval; the trust radius
    /// should be clipped before retraction.
    IntervalNearBoundary,
}

impl ManifoldWarning {
    /// Stable string representation (for ad-hoc logging).
    pub const fn as_str(self) -> &'static str {
        match self {
            ManifoldWarning::SphereNearPole => {
                "sphere: near chart pole, retraction may amplify error"
            }
            ManifoldWarning::IntervalNearBoundary => {
                "interval: near boundary; trust radius should be clipped"
            }
        }
    }
}

/// Trait for a smooth manifold on which a per-point latent coordinate lives.
///
/// Conventions:
///   * `p` is the *embedding* point (length = ambient dimension).
///   * `xi` / `eta` are tangent vectors at `p` in the *embedding* coordinates
///     (same length as `p`); intrinsic-dimension coordinates are not exposed
///     here. This is the Pymanopt / Manopt convention.
///   * [`Self::dim`] returns the *intrinsic* manifold dimension.
pub trait Manifold: Send + Sync {
    /// Intrinsic manifold dimension.
    fn dim(&self) -> usize;

    /// Ambient (embedding) dimension. For [`Euclidean`] this equals `dim`;
    /// for [`Circle`] this is `2`; for [`Sphere`] of S^n this is `n+1`.
    fn ambient_dim(&self) -> usize;

    /// Project a vector `v` onto the tangent space `T_p M` in place.
    fn project_tangent(&self, p: ArrayView1<f64>, v: ArrayViewMut1<f64>);

    /// Retract `p + ξ` back onto the manifold: writes `R_p(ξ)` to `out`.
    fn retract(&self, p: ArrayView1<f64>, xi: ArrayView1<f64>, out: ArrayViewMut1<f64>);

    /// Transport tangent `xi` at `from` to a tangent vector at `to` in place
    /// (projection approximation — not full parallel transport).
    fn vector_transport(
        &self,
        from: ArrayView1<f64>,
        to: ArrayView1<f64>,
        xi: ArrayViewMut1<f64>,
    );

    /// Per-ambient-axis trust-region metric weights.
    fn metric_weights(&self) -> Vec<f64> {
        vec![1.0; self.ambient_dim()]
    }

    /// Riemannian inner product `<ξ, η>_p`. Default: weighted ambient
    /// inner product restricted to `T_p M`.
    fn inner_product(&self, _p: ArrayView1<f64>, xi: ArrayView1<f64>, eta: ArrayView1<f64>)
        -> f64 {
        debug_assert_eq!(xi.len(), eta.len());
        let weights = self.metric_weights();
        debug_assert_eq!(weights.len(), xi.len());
        let mut acc = 0.0_f64;
        for i in 0..xi.len() {
            acc += weights[i] * xi[i] * eta[i];
        }
        acc
    }

    /// Convert a Euclidean gradient to a Riemannian gradient in place:
    /// `grad_R = P_{T_p M}(grad_E)`.
    fn euclidean_to_riemannian_grad(&self, p: ArrayView1<f64>, egrad: ArrayViewMut1<f64>) {
        self.project_tangent(p, egrad);
    }

    /// Convert a Euclidean Hessian-vector product to a Riemannian one:
    /// `Hess_R[ξ] = P_{T_p M}(Hess_E[ξ] - W_p(ξ, grad_E))`
    /// where `W_p` is the Weingarten / shape operator.
    fn euclidean_to_riemannian_hess_vp(
        &self,
        p: ArrayView1<f64>,
        egrad: ArrayView1<f64>,
        ehess_vp: ArrayViewMut1<f64>,
        xi: ArrayView1<f64>,
    );

    /// Diagnostic name (for logs / errors).
    fn name(&self) -> &str;

    /// Optional runtime advisory: callers may inspect this and log when the
    /// point is too close to a chart singularity. Default returns `None`.
    fn warn_at(&self, p: ArrayView1<f64>) -> Option<&'static str> {
        self.warn_at_typed(p).map(ManifoldWarning::as_str)
    }

    /// Typed advisory sentinel — preferred over [`Manifold::warn_at`] for
    /// callers that route to a structured logger. Default returns `None`.
    fn warn_at_typed(&self, _p: ArrayView1<f64>) -> Option<ManifoldWarning> {
        None
    }

    /// Orthonormal basis `Q ∈ ℝ^{m × d}` for the tangent space `T_p M`,
    /// where `m = ambient_dim()` and `d = dim()`.
    ///
    /// Default implementation: project each ambient basis vector `e_j` onto
    /// `T_p M` and orthonormalize the resulting `m × m` matrix via modified
    /// Gram-Schmidt, keeping the first `dim()` columns whose norm exceeds a
    /// numerical tolerance. This is generic and correct for every shipped
    /// manifold; concrete manifolds can override with a closed-form basis if
    /// it is cheaper (the default is `O(m² · dim) ≤ O(m³)`).
    ///
    /// For [`Euclidean`] (`m = d`) the returned matrix is the identity, so
    /// solving in the tangent basis is bit-equivalent to solving in the
    /// ambient space.
    fn tangent_basis(&self, p: ArrayView1<f64>) -> Array2<f64> {
        let m = self.ambient_dim();
        let d = self.dim();
        // Project each ambient basis vector onto T_p M.
        let mut cols: Vec<Array1<f64>> = Vec::with_capacity(m);
        for j in 0..m {
            let mut e = Array1::<f64>::zeros(m);
            e[j] = 1.0;
            self.project_tangent(p, e.view_mut());
            cols.push(e);
        }
        // Modified Gram–Schmidt; keep up to `d` orthonormal columns.
        let tol = 1.0e-12;
        let mut basis: Vec<Array1<f64>> = Vec::with_capacity(d);
        for mut v in cols.into_iter() {
            if basis.len() == d {
                break;
            }
            for q in basis.iter() {
                let mut dot = 0.0_f64;
                for i in 0..m {
                    dot += v[i] * q[i];
                }
                for i in 0..m {
                    v[i] -= dot * q[i];
                }
            }
            let mut nrm2 = 0.0_f64;
            for i in 0..m {
                nrm2 += v[i] * v[i];
            }
            let nrm = nrm2.sqrt();
            if nrm > tol {
                for i in 0..m {
                    v[i] /= nrm;
                }
                basis.push(v);
            }
        }
        let cols_kept = basis.len();
        let mut q = Array2::<f64>::zeros((m, cols_kept));
        for (j, col) in basis.into_iter().enumerate() {
            for i in 0..m {
                q[[i, j]] = col[i];
            }
        }
        q
    }
}

// ---------------------------------------------------------------------------
// Concrete manifolds
// ---------------------------------------------------------------------------

/// Plain Euclidean ℝ^d. All operations are identity / no-op so any caller
/// that opts in to `Manifold::Euclidean` sees bit-equivalent behaviour to
/// the pre-Riemannian solver path.
#[derive(Debug, Clone)]
pub struct Euclidean {
    pub d: usize,
}

impl Manifold for Euclidean {
    fn dim(&self) -> usize {
        self.d
    }
    fn ambient_dim(&self) -> usize {
        self.d
    }
    fn project_tangent(&self, _p: ArrayView1<f64>, _v: ArrayViewMut1<f64>) {}
    fn retract(&self, p: ArrayView1<f64>, xi: ArrayView1<f64>, mut out: ArrayViewMut1<f64>) {
        debug_assert_eq!(p.len(), self.d);
        debug_assert_eq!(xi.len(), self.d);
        for i in 0..self.d {
            out[i] = p[i] + xi[i];
        }
    }
    fn vector_transport(
        &self,
        _from: ArrayView1<f64>,
        _to: ArrayView1<f64>,
        _xi: ArrayViewMut1<f64>,
    ) {
    }
    fn euclidean_to_riemannian_hess_vp(
        &self,
        _p: ArrayView1<f64>,
        _egrad: ArrayView1<f64>,
        _ehess_vp: ArrayViewMut1<f64>,
        _xi: ArrayView1<f64>,
    ) {
        // Identity: no Weingarten correction in flat space.
    }
    fn name(&self) -> &str {
        "Euclidean"
    }
}

/// The unit circle S¹ embedded in ℝ² as `(cos θ, sin θ)`. Tangent at
/// `p = (x, y)` is the line spanned by `(-y, x)`.
#[derive(Debug, Clone)]
pub struct Circle;

impl Manifold for Circle {
    fn dim(&self) -> usize {
        1
    }
    fn ambient_dim(&self) -> usize {
        2
    }
    fn metric_weights(&self) -> Vec<f64> {
        let w = 1.0 / (TWO_PI * TWO_PI);
        vec![w; 2]
    }
    fn project_tangent(&self, p: ArrayView1<f64>, mut v: ArrayViewMut1<f64>) {
        debug_assert_eq!(p.len(), 2);
        debug_assert_eq!(v.len(), 2);
        // P_p(v) = v - <p,v> p ;  idempotent: P_p ∘ P_p = P_p.
        let dot = v[0] * p[0] + v[1] * p[1];
        v[0] -= dot * p[0];
        v[1] -= dot * p[1];
        // Tangency invariant: <p, v> ≈ 0 after one application.
        debug_assert!((v[0] * p[0] + v[1] * p[1]).abs() < 1.0e-9);
    }
    fn retract(&self, p: ArrayView1<f64>, xi: ArrayView1<f64>, mut out: ArrayViewMut1<f64>) {
        // Closed-form projective retraction (proposal §4.2):
        //   R_p(ξ) = (p + ξ)/||p + ξ||.
        // First-order accurate, seam-free, no angle parameterization needed
        // (so no 2π wrap to handle here — periodicity is automatic).
        let x = p[0] + xi[0];
        let y = p[1] + xi[1];
        let s2 = x * x + y * y;
        debug_assert!(s2.is_finite() && s2 > 0.0, "Circle::retract degenerate ||p+ξ||");
        let norm = s2.sqrt().max(1.0e-300);
        out[0] = x / norm;
        out[1] = y / norm;
        // Post-condition: retraction stays on S¹.
        debug_assert!((out[0] * out[0] + out[1] * out[1] - 1.0).abs() < 1.0e-9);
    }
    fn vector_transport(
        &self,
        _from: ArrayView1<f64>,
        to: ArrayView1<f64>,
        xi: ArrayViewMut1<f64>,
    ) {
        // Projection approximation τ_{p→q}(ξ) = P_q(ξ) (proposal §4.4).
        // Safe even at antipodal endpoints since it does not divide by 1+<p,q>.
        self.project_tangent(to, xi);
    }
    fn euclidean_to_riemannian_hess_vp(
        &self,
        p: ArrayView1<f64>,
        egrad: ArrayView1<f64>,
        mut ehess_vp: ArrayViewMut1<f64>,
        xi: ArrayView1<f64>,
    ) {
        // S¹ Weingarten (proposal §4.7 / §18):
        //   W_p(ξ, v) = <p, v> ξ
        // Hess_R[ξ] = P_p(∇²_E · ξ - W_p(ξ, ∇_E)).
        // CRITICAL: the second argument of W is the *ambient* (or normal)
        // gradient ∇_E, not P_p(∇_E). If we passed the projected gradient
        // here the radial-egrad scalar would be exactly zero and the
        // curvature correction would silently vanish.
        debug_assert_eq!(p.len(), 2);
        debug_assert_eq!(egrad.len(), 2);
        debug_assert_eq!(ehess_vp.len(), 2);
        debug_assert_eq!(xi.len(), 2);
        let radial_egrad = egrad[0] * p[0] + egrad[1] * p[1];
        ehess_vp[0] -= radial_egrad * xi[0];
        ehess_vp[1] -= radial_egrad * xi[1];
        self.project_tangent(p, ehess_vp);
    }
    /// Closed-form tangent basis: a single unit vector orthogonal to `p`,
    /// `Q(p) = [-p_y, p_x]ᵀ` (proposal §4.1). O(1) vs the default O(m³).
    fn tangent_basis(&self, p: ArrayView1<f64>) -> Array2<f64> {
        debug_assert_eq!(p.len(), 2);
        let mut q = Array2::<f64>::zeros((2, 1));
        q[[0, 0]] = -p[1];
        q[[1, 0]] = p[0];
        // p is already unit on the manifold, so ||Q|| = 1 by construction;
        // we do not renormalize here.
        q
    }
    fn name(&self) -> &str {
        "Circle"
    }
}

/// The unit n-sphere S^n embedded in ℝ^{n+1} as a unit vector.
#[derive(Debug, Clone)]
pub struct Sphere {
    /// Intrinsic dimension; ambient = n + 1.
    pub n: usize,
}

impl Manifold for Sphere {
    fn dim(&self) -> usize {
        self.n
    }
    fn ambient_dim(&self) -> usize {
        self.n + 1
    }
    fn metric_weights(&self) -> Vec<f64> {
        let w = 1.0 / (std::f64::consts::PI * std::f64::consts::PI);
        vec![w; self.n + 1]
    }
    fn project_tangent(&self, p: ArrayView1<f64>, mut v: ArrayViewMut1<f64>) {
        debug_assert_eq!(p.len(), self.n + 1);
        debug_assert_eq!(v.len(), self.n + 1);
        // P_p(v) = (I - pp^T) v = v - <p, v> p (proposal §6.3). Idempotent.
        let mut dot = 0.0_f64;
        for i in 0..p.len() {
            dot += v[i] * p[i];
        }
        for i in 0..p.len() {
            v[i] -= dot * p[i];
        }
        if cfg!(debug_assertions) {
            let mut chk = 0.0_f64;
            for i in 0..p.len() {
                chk += v[i] * p[i];
            }
            debug_assert!(chk.abs() < 1.0e-8 * (1.0 + p.len() as f64));
        }
    }
    fn retract(&self, p: ArrayView1<f64>, xi: ArrayView1<f64>, mut out: ArrayViewMut1<f64>) {
        // Closed-form projective retraction R_p(ξ) = (p + ξ)/||p + ξ||
        // (proposal §6.2). Caller is expected to have already projected ξ
        // onto T_p M and clipped its trust radius.
        let m = self.n + 1;
        let mut s2 = 0.0_f64;
        for i in 0..m {
            let v = p[i] + xi[i];
            out[i] = v;
            s2 += v * v;
        }
        debug_assert!(s2.is_finite() && s2 > 0.0, "Sphere::retract degenerate ||p+ξ||");
        let norm = s2.sqrt().max(1.0e-300);
        for i in 0..m {
            out[i] /= norm;
        }
        if cfg!(debug_assertions) {
            let mut n2 = 0.0_f64;
            for i in 0..m {
                n2 += out[i] * out[i];
            }
            debug_assert!((n2 - 1.0).abs() < 1.0e-9, "Sphere::retract output not on S^n");
        }
    }
    fn vector_transport(
        &self,
        _from: ArrayView1<f64>,
        to: ArrayView1<f64>,
        xi: ArrayViewMut1<f64>,
    ) {
        // Projection transport (proposal §6.4). Cheap, stable, not exactly
        // isometric. Antipodal case (to ≈ -from) does not divide by zero
        // here — unlike exact geodesic transport which has a 1+<p,q>
        // denominator. The transported tangent can be small near antipodal
        // endpoints; that is a correctness-preserving degradation.
        self.project_tangent(to, xi);
    }
    fn euclidean_to_riemannian_hess_vp(
        &self,
        p: ArrayView1<f64>,
        egrad: ArrayView1<f64>,
        mut ehess_vp: ArrayViewMut1<f64>,
        xi: ArrayView1<f64>,
    ) {
        // Proposal §6.7 / §18:  W_p(ξ, v) = <p, v> ξ for the unit sphere.
        // Hess_R[ξ] = P_p(∇²_E · ξ - W_p(ξ, ∇_E)).
        // The second arg of W is the AMBIENT gradient ∇_E. Passing the
        // projected gradient would make <p, ∇_E> ≡ 0 and silently drop the
        // curvature correction — this matches the explicit warning in the
        // derivation.
        debug_assert_eq!(p.len(), self.n + 1);
        debug_assert_eq!(egrad.len(), self.n + 1);
        debug_assert_eq!(ehess_vp.len(), self.n + 1);
        debug_assert_eq!(xi.len(), self.n + 1);
        let mut radial_egrad = 0.0_f64;
        for i in 0..p.len() {
            radial_egrad += egrad[i] * p[i];
        }
        for i in 0..ehess_vp.len() {
            ehess_vp[i] -= radial_egrad * xi[i];
        }
        self.project_tangent(p, ehess_vp);
    }
    /// Closed-form orthonormal tangent basis via Householder reflection:
    /// map `e_m → p`, then drop the column that would span `p`. This is
    /// O(m) vs the default modified-Gram-Schmidt O(m^3).
    ///
    /// Construction: choose anchor `a = e_k` with `k = argmax |p_i|` for
    /// numerical stability. Householder vector `u = (p - a)/||p - a||`
    /// reflects `e_k` to ±p. The reflection `H = I - 2 u u^T` sends `e_k`
    /// to `±p`; the remaining `m-1` ambient basis vectors map to an
    /// orthonormal basis of `T_p M`.
    fn tangent_basis(&self, p: ArrayView1<f64>) -> Array2<f64> {
        let m = self.n + 1;
        debug_assert_eq!(p.len(), m);
        if m == 0 {
            return Array2::<f64>::zeros((0, 0));
        }
        // Pick the largest-|p_i| as the Householder anchor axis for stability.
        let mut anchor = 0usize;
        let mut amax = -1.0_f64;
        for i in 0..m {
            let a = p[i].abs();
            if a > amax {
                amax = a;
                anchor = i;
            }
        }
        // Reflect e_anchor onto sign(p_anchor) * p so 1 + <e_anchor, ...>·sign
        // stays bounded away from zero (avoids the antipodal degeneracy).
        let sign = if p[anchor] >= 0.0 { 1.0 } else { -1.0 };
        // u = p - sign * e_anchor (then renormalize).
        let mut u = Array1::<f64>::zeros(m);
        for i in 0..m {
            u[i] = sign * p[i];
        }
        u[anchor] -= 1.0;
        let mut u_n2 = 0.0_f64;
        for i in 0..m {
            u_n2 += u[i] * u[i];
        }
        // If u is essentially zero, p ≈ sign·e_anchor exactly; the natural
        // tangent basis is just the other axes.
        let mut basis = Array2::<f64>::zeros((m, self.n));
        if u_n2 < 1.0e-30 {
            let mut col = 0usize;
            for i in 0..m {
                if i == anchor {
                    continue;
                }
                basis[[i, col]] = 1.0;
                col += 1;
            }
            return basis;
        }
        let inv_un = 1.0 / u_n2.sqrt();
        for i in 0..m {
            u[i] *= inv_un;
        }
        // Columns of basis are H · e_j for j ≠ anchor, where H = I - 2 u u^T.
        // H · e_j = e_j - 2 u_j u.
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
        basis
    }
    fn warn_at_typed(&self, p: ArrayView1<f64>) -> Option<ManifoldWarning> {
        // Advisory only: if the last coordinate is very small the canonical
        // chart near the equator/pole is ill-conditioned.
        if let Some(last) = p.iter().last() {
            if last.abs() < SPHERE_POLE_WARN_THRESHOLD {
                return Some(ManifoldWarning::SphereNearPole);
            }
        }
        None
    }
    fn name(&self) -> &str {
        "Sphere"
    }
}

/// Closed interval `[lo, hi]` parameterised smoothly by `tanh`:
///   `p = (lo + hi)/2 + (hi - lo)/2 · tanh(z)`.
/// The retraction here works in the natural coordinate (`p` directly) but
/// clips to a band `[lo + ε, hi − ε]` to keep the Jacobian of the implicit
/// `tanh` chart bounded.
#[derive(Debug, Clone)]
pub struct Interval {
    pub lo: f64,
    pub hi: f64,
}

impl Interval {
    /// Edge band fraction kept clear of the boundary.
    const EDGE_FRAC: f64 = 1.0e-6;
    fn clip(&self, x: f64) -> f64 {
        let band = (self.hi - self.lo).abs() * Self::EDGE_FRAC;
        x.max(self.lo + band).min(self.hi - band)
    }
}

impl Manifold for Interval {
    fn dim(&self) -> usize {
        1
    }
    fn ambient_dim(&self) -> usize {
        1
    }
    fn metric_weights(&self) -> Vec<f64> {
        let scale = self.hi - self.lo;
        vec![1.0 / (scale * scale)]
    }
    fn project_tangent(&self, _p: ArrayView1<f64>, _v: ArrayViewMut1<f64>) {
        // 1-d open submanifold of ℝ; tangent space is all of ℝ.
    }
    fn retract(&self, p: ArrayView1<f64>, xi: ArrayView1<f64>, mut out: ArrayViewMut1<f64>) {
        // Closed-constraint retraction (proposal §7.2):
        //   R_p(ξ) = clamp(p + ξ, lo+ε, hi−ε).
        // The clip lives in the *ambient* (= internal) coordinate before any
        // tanh chart would be applied, so the implicit chart Jacobian
        // J = r·(1 - tanh(z)^2) stays bounded throughout the optimizer
        // (this matches proposal §15 numerical-pitfalls discussion).
        debug_assert_eq!(p.len(), 1);
        debug_assert_eq!(xi.len(), 1);
        debug_assert_eq!(out.len(), 1);
        out[0] = self.clip(p[0] + xi[0]);
        debug_assert!(
            out[0] > self.lo && out[0] < self.hi,
            "Interval::retract output left feasible band"
        );
    }
    fn vector_transport(
        &self,
        _from: ArrayView1<f64>,
        _to: ArrayView1<f64>,
        _xi: ArrayViewMut1<f64>,
    ) {
    }
    fn euclidean_to_riemannian_hess_vp(
        &self,
        _p: ArrayView1<f64>,
        _egrad: ArrayView1<f64>,
        _ehess_vp: ArrayViewMut1<f64>,
        _xi: ArrayView1<f64>,
    ) {
        // Open submanifold of ℝ: no second fundamental form correction in
        // the natural chart used here.
    }
    fn warn_at_typed(&self, p: ArrayView1<f64>) -> Option<ManifoldWarning> {
        let band = (self.hi - self.lo).abs() * Self::EDGE_FRAC * 10.0;
        if p[0] < self.lo + band || p[0] > self.hi - band {
            Some(ManifoldWarning::IntervalNearBoundary)
        } else {
            None
        }
    }
    /// Closed-form trivial 1×1 identity basis (proposal §7: T_p M = ℝ in
    /// the interior). Override avoids the generic O(m³) Gram-Schmidt path.
    fn tangent_basis(&self, _p: ArrayView1<f64>) -> Array2<f64> {
        let mut q = Array2::<f64>::zeros((1, 1));
        q[[0, 0]] = 1.0;
        q
    }
    fn name(&self) -> &str {
        "Interval"
    }
}

/// Torus T^d = (S¹)^d. Embedding is the concatenation of `d` 2-vectors;
/// ambient dimension is `2 * d`.
#[derive(Debug, Clone)]
pub struct Torus {
    pub d: usize,
}

impl Manifold for Torus {
    fn dim(&self) -> usize {
        self.d
    }
    fn ambient_dim(&self) -> usize {
        2 * self.d
    }
    fn metric_weights(&self) -> Vec<f64> {
        let w = 1.0 / (TWO_PI * TWO_PI);
        vec![w; 2 * self.d]
    }
    fn project_tangent(&self, p: ArrayView1<f64>, mut v: ArrayViewMut1<f64>) {
        for k in 0..self.d {
            let px = p[2 * k];
            let py = p[2 * k + 1];
            let dot = v[2 * k] * px + v[2 * k + 1] * py;
            v[2 * k] -= dot * px;
            v[2 * k + 1] -= dot * py;
        }
    }
    fn retract(&self, p: ArrayView1<f64>, xi: ArrayView1<f64>, mut out: ArrayViewMut1<f64>) {
        // Blockwise S¹ retraction (proposal §9.2):
        //   R_p(ξ)_j = (p_j + ξ_j)/||p_j + ξ_j||.
        // Per-axis factoring keeps the cost O(d) instead of touching cross
        // terms (which the curvature correction also avoids).
        for k in 0..self.d {
            let x = p[2 * k] + xi[2 * k];
            let y = p[2 * k + 1] + xi[2 * k + 1];
            let s2 = x * x + y * y;
            debug_assert!(s2.is_finite() && s2 > 0.0, "Torus::retract degenerate at axis {}", k);
            let norm = s2.sqrt().max(1.0e-300);
            out[2 * k] = x / norm;
            out[2 * k + 1] = y / norm;
            if cfg!(debug_assertions) {
                let n2 = out[2 * k] * out[2 * k] + out[2 * k + 1] * out[2 * k + 1];
                debug_assert!((n2 - 1.0).abs() < 1.0e-9);
            }
        }
    }
    fn vector_transport(
        &self,
        _from: ArrayView1<f64>,
        to: ArrayView1<f64>,
        xi: ArrayViewMut1<f64>,
    ) {
        self.project_tangent(to, xi);
    }
    fn euclidean_to_riemannian_hess_vp(
        &self,
        p: ArrayView1<f64>,
        egrad: ArrayView1<f64>,
        mut ehess_vp: ArrayViewMut1<f64>,
        xi: ArrayView1<f64>,
    ) {
        // Blockwise S¹ Weingarten on each circle factor (proposal §9.7 / §18):
        //   W_p(ξ, v)_j = <p_j, v_j> ξ_j.
        // The projection P_p is itself block-diagonal so the *full* product
        // structure here is per-axis; we never need to mix cross-axis entries
        // of the ambient Hessian for the curvature correction (only ∇²_E·ξ
        // — supplied by the caller — may legitimately have cross-axis
        // entries, and we keep them through the projection step).
        for k in 0..self.d {
            let radial = egrad[2 * k] * p[2 * k] + egrad[2 * k + 1] * p[2 * k + 1];
            ehess_vp[2 * k] -= radial * xi[2 * k];
            ehess_vp[2 * k + 1] -= radial * xi[2 * k + 1];
        }
        self.project_tangent(p, ehess_vp);
    }
    /// Closed-form block-diagonal basis (proposal §11):
    ///   Q = blockdiag(Q_1, ..., Q_d) with Q_j = [-p_{j,y}, p_{j,x}]ᵀ.
    /// O(d) vs default O(m³) where m = 2d.
    fn tangent_basis(&self, p: ArrayView1<f64>) -> Array2<f64> {
        debug_assert_eq!(p.len(), 2 * self.d);
        let mut q = Array2::<f64>::zeros((2 * self.d, self.d));
        for k in 0..self.d {
            q[[2 * k, k]] = -p[2 * k + 1];
            q[[2 * k + 1, k]] = p[2 * k];
        }
        q
    }
    fn name(&self) -> &str {
        "Torus"
    }
}

/// Cartesian product of manifolds; ambient dimensions concatenate.
pub struct Product {
    pub components: Vec<Box<dyn Manifold>>,
    pub weights: Option<Vec<f64>>,
}

impl Manifold for Product {
    fn dim(&self) -> usize {
        self.components.iter().map(|c| c.dim()).sum()
    }
    fn ambient_dim(&self) -> usize {
        self.components.iter().map(|c| c.ambient_dim()).sum()
    }
    fn metric_weights(&self) -> Vec<f64> {
        if let Some(weights) = &self.weights {
            assert_eq!(
                weights.len(),
                self.ambient_dim(),
                "Product manifold metric weights length must match ambient dimension"
            );
            return weights.clone();
        }
        let mut out = Vec::with_capacity(self.ambient_dim());
        for c in &self.components {
            out.extend(c.metric_weights());
        }
        out
    }
    fn project_tangent(&self, p: ArrayView1<f64>, v: ArrayViewMut1<f64>) {
        let mut off = 0usize;
        let mut v_mut = v;
        for c in &self.components {
            let m = c.ambient_dim();
            let p_slice = p.slice(ndarray::s![off..off + m]);
            let v_slice = v_mut.slice_mut(ndarray::s![off..off + m]);
            c.project_tangent(p_slice, v_slice);
            off += m;
        }
    }
    fn retract(&self, p: ArrayView1<f64>, xi: ArrayView1<f64>, out: ArrayViewMut1<f64>) {
        let mut off = 0usize;
        let mut out_mut = out;
        for c in &self.components {
            let m = c.ambient_dim();
            let p_slice = p.slice(ndarray::s![off..off + m]);
            let xi_slice = xi.slice(ndarray::s![off..off + m]);
            let out_slice = out_mut.slice_mut(ndarray::s![off..off + m]);
            c.retract(p_slice, xi_slice, out_slice);
            off += m;
        }
    }
    fn vector_transport(
        &self,
        from: ArrayView1<f64>,
        to: ArrayView1<f64>,
        xi: ArrayViewMut1<f64>,
    ) {
        let mut off = 0usize;
        let mut xi_mut = xi;
        for c in &self.components {
            let m = c.ambient_dim();
            let from_slice = from.slice(ndarray::s![off..off + m]);
            let to_slice = to.slice(ndarray::s![off..off + m]);
            let xi_slice = xi_mut.slice_mut(ndarray::s![off..off + m]);
            c.vector_transport(from_slice, to_slice, xi_slice);
            off += m;
        }
    }
    fn euclidean_to_riemannian_hess_vp(
        &self,
        p: ArrayView1<f64>,
        egrad: ArrayView1<f64>,
        ehess_vp: ArrayViewMut1<f64>,
        xi: ArrayView1<f64>,
    ) {
        let mut off = 0usize;
        let mut ehess_mut = ehess_vp;
        for c in &self.components {
            let m = c.ambient_dim();
            let p_slice = p.slice(ndarray::s![off..off + m]);
            let eg_slice = egrad.slice(ndarray::s![off..off + m]);
            let xi_slice = xi.slice(ndarray::s![off..off + m]);
            let eh_slice = ehess_mut.slice_mut(ndarray::s![off..off + m]);
            c.euclidean_to_riemannian_hess_vp(p_slice, eg_slice, eh_slice, xi_slice);
            off += m;
        }
    }
    fn name(&self) -> &str {
        "Product"
    }
}

// ---------------------------------------------------------------------------
// Serialization-friendly config
// ---------------------------------------------------------------------------

/// Serialization-friendly description of a manifold (no trait objects).
#[derive(Debug, Clone)]
pub enum ManifoldKind {
    /// Flat Euclidean ℝ^d. Default; bit-equivalent to no Riemannian step.
    Euclidean(usize),
    /// Unit circle S¹.
    Circle,
    /// Unit n-sphere S^n.
    Sphere(usize),
    /// Closed interval `[lo, hi]`.
    Interval(f64, f64),
    /// Torus T^d = (S¹)^d.
    Torus(usize),
    /// Cartesian product.
    Product(Vec<ManifoldKind>),
    /// Cartesian product with explicit per-ambient-axis metric weights.
    ProductWithMetric {
        components: Vec<ManifoldKind>,
        weights: Vec<f64>,
    },
}

impl ManifoldKind {
    /// Materialise a trait-object instance.
    #[must_use]
    pub fn build(&self) -> Box<dyn Manifold> {
        match self {
            ManifoldKind::Euclidean(d) => Box::new(Euclidean { d: *d }),
            ManifoldKind::Circle => Box::new(Circle),
            ManifoldKind::Sphere(n) => Box::new(Sphere { n: *n }),
            ManifoldKind::Interval(lo, hi) => Box::new(Interval { lo: *lo, hi: *hi }),
            ManifoldKind::Torus(d) => Box::new(Torus { d: *d }),
            ManifoldKind::Product(components) => Box::new(Product {
                components: components.iter().map(|c| c.build()).collect(),
                weights: None,
            }),
            ManifoldKind::ProductWithMetric {
                components,
                weights,
            } => Box::new(Product {
                components: components.iter().map(|c| c.build()).collect(),
                weights: Some(weights.clone()),
            }),
        }
    }

    /// Returns `true` iff this manifold is bit-equivalent to no-op flat space.
    pub fn is_euclidean(&self) -> bool {
        matches!(self, ManifoldKind::Euclidean(_))
    }

    /// Ambient dimension implied by this manifold spec.
    pub fn ambient_dim(&self) -> usize {
        match self {
            ManifoldKind::Euclidean(d) => *d,
            ManifoldKind::Circle => 2,
            ManifoldKind::Sphere(n) => n + 1,
            ManifoldKind::Interval(_, _) => 1,
            ManifoldKind::Torus(d) => 2 * d,
            ManifoldKind::Product(components)
            | ManifoldKind::ProductWithMetric {
                components,
                weights: _,
            } => components.iter().map(|c| c.ambient_dim()).sum(),
        }
    }
}

// ---------------------------------------------------------------------------
// Convenience: apply a Euclidean delta as a Riemannian retraction step.
// ---------------------------------------------------------------------------

/// Treat `delta` (length = ambient_dim) as a Euclidean tangent vector at
/// `point`, project it to the tangent space, and retract. Writes the new
/// point to `out_new_point`.
///
/// This is the minimal hook used by `arrow_schur` and `persistent_warm_start`
/// to wrap an existing Euclidean increment through a retraction without
/// re-deriving the Newton step. For `ManifoldKind::Euclidean` this collapses
/// to `out = point + delta` (bit-equivalent to the current code path).
pub fn retract_euclidean_delta(
    manifold: &dyn Manifold,
    point: ArrayView1<f64>,
    delta: ArrayView1<f64>,
    out_new_point: ArrayViewMut1<f64>,
) {
    let m = manifold.ambient_dim();
    debug_assert_eq!(point.len(), m);
    debug_assert_eq!(delta.len(), m);
    let mut xi = delta.to_owned();
    manifold.project_tangent(point, xi.view_mut());
    manifold.retract(point, xi.view(), out_new_point);
}

// ===========================================================================
// Tests (build-only — invoked only by `cargo check --all-targets`)
// ===========================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::array;

    fn norm2(v: ArrayView1<f64>) -> f64 {
        v.iter().map(|x| x * x).sum::<f64>().sqrt()
    }

    #[test]
    fn circle_retraction_stays_unit() {
        let m = Circle;
        let p = array![1.0_f64, 0.0];
        let xi = array![0.0_f64, 0.3];
        let mut out = array![0.0_f64, 0.0];
        m.retract(p.view(), xi.view(), out.view_mut());
        let n = norm2(out.view());
        assert!((n - 1.0).abs() < 1.0e-12);
    }

    #[test]
    fn sphere_tangent_orthogonal_to_point() {
        let m = Sphere { n: 2 };
        // p = (0,0,1)
        let p = array![0.0_f64, 0.0, 1.0];
        let mut v = array![1.0_f64, 2.0, 3.0];
        m.project_tangent(p.view(), v.view_mut());
        let dot: f64 = (0..3).map(|i| v[i] * p[i]).sum();
        assert!(dot.abs() < 1.0e-12);
    }

    #[test]
    fn interval_stays_strictly_inside() {
        let m = Interval { lo: -1.0, hi: 1.0 };
        let p = array![0.99_f64];
        let xi = array![10.0_f64];
        let mut out = array![0.0_f64];
        m.retract(p.view(), xi.view(), out.view_mut());
        assert!(out[0] > -1.0 && out[0] < 1.0);
    }

    #[test]
    fn euclidean_is_identity() {
        let m = Euclidean { d: 3 };
        let p = array![0.1_f64, 0.2, 0.3];
        let xi = array![1.0_f64, -1.0, 0.5];
        let mut out = array![0.0_f64, 0.0, 0.0];
        m.retract(p.view(), xi.view(), out.view_mut());
        for i in 0..3 {
            assert!((out[i] - (p[i] + xi[i])).abs() < 1.0e-15);
        }
    }

    #[test]
    fn weingarten_correction_matches_two_paths_on_sphere() {
        // For a sphere with Euclidean Hessian H_E and gradient g, the
        // Riemannian Hess applied to a tangent vector ξ should equal
        //   P_T( H_E ξ - <p, g> ξ ).
        let m = Sphere { n: 2 };
        let p = array![1.0_f64 / 3.0_f64.sqrt(), 1.0 / 3.0_f64.sqrt(), 1.0 / 3.0_f64.sqrt()];
        let egrad = array![0.5_f64, -0.2, 0.7];
        // Tangent: cross with arbitrary direction then re-project.
        let mut xi = array![1.0_f64, 0.0, 0.0];
        m.project_tangent(p.view(), xi.view_mut());

        // Path A: via euclidean_to_riemannian_hess_vp using identity ehess.
        let mut path_a = xi.clone(); // identity H_E
        m.euclidean_to_riemannian_hess_vp(p.view(), egrad.view(), path_a.view_mut(), xi.view());

        // Path B: manual.
        let radial: f64 = (0..3).map(|i| egrad[i] * p[i]).sum();
        let mut path_b = xi.clone();
        for i in 0..3 {
            path_b[i] -= radial * xi[i];
        }
        m.project_tangent(p.view(), path_b.view_mut());

        for i in 0..3 {
            assert!(
                (path_a[i] - path_b[i]).abs() < 1.0e-12,
                "weingarten mismatch at {i}: {} vs {}",
                path_a[i],
                path_b[i]
            );
        }
    }
}
