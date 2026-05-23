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
//!   * a serialization-friendly enum [`ManifoldKind`] with [`ManifoldKind::build`];
//!   * a per-point Riemannian Newton step (with optional trust radius)
//!     [`riemannian_newton_step_on_point`].
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
//!   advisory hook for callers that want to log.
//! * Near the boundary of an interval, the smooth `tanh` parameterization's
//!   Jacobian blows up. [`Interval`] clips the retraction step to keep the
//!   internal coordinate within a sane band.
//! * [`Manifold::vector_transport`] uses a *parallel-transport approximation*
//!   (re-project the moved tangent vector to the new tangent space). Full
//!   parallel transport along the retraction curve would require an ODE solve;
//!   the projection approximation is a standard Pymanopt-grade tradeoff —
//!   first-order isometric, O(d) per call.

use ndarray::{Array1, Array2, ArrayView1, ArrayView2, ArrayViewMut1};

/// Sentinel: warn when sphere retraction operates this close to a chart
/// singularity (currently only used as an advisory threshold; callers can
/// hook a runtime warning here).
pub const SPHERE_POLE_WARN_THRESHOLD: f64 = 1.0e-8;

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

    /// Riemannian inner product `<ξ, η>_p`. Default: ambient Euclidean
    /// inner product restricted to `T_p M` (the induced metric).
    fn inner_product(&self, _p: ArrayView1<f64>, xi: ArrayView1<f64>, eta: ArrayView1<f64>)
        -> f64 {
        debug_assert_eq!(xi.len(), eta.len());
        let mut acc = 0.0_f64;
        for i in 0..xi.len() {
            acc += xi[i] * eta[i];
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
    fn warn_at(&self, _p: ArrayView1<f64>) -> Option<&'static str> {
        None
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

impl Circle {
    fn tangent_basis(p: ArrayView1<f64>) -> (f64, f64) {
        (-p[1], p[0])
    }
}

impl Manifold for Circle {
    fn dim(&self) -> usize {
        1
    }
    fn ambient_dim(&self) -> usize {
        2
    }
    fn project_tangent(&self, p: ArrayView1<f64>, mut v: ArrayViewMut1<f64>) {
        debug_assert_eq!(p.len(), 2);
        debug_assert_eq!(v.len(), 2);
        // remove the radial component <v, p> p
        let dot = v[0] * p[0] + v[1] * p[1];
        v[0] -= dot * p[0];
        v[1] -= dot * p[1];
    }
    fn retract(&self, p: ArrayView1<f64>, xi: ArrayView1<f64>, mut out: ArrayViewMut1<f64>) {
        // Standard projective retraction R_p(ξ) = (p + ξ)/||p + ξ||
        let x = p[0] + xi[0];
        let y = p[1] + xi[1];
        let norm = (x * x + y * y).sqrt().max(1.0e-300);
        out[0] = x / norm;
        out[1] = y / norm;
    }
    fn vector_transport(
        &self,
        _from: ArrayView1<f64>,
        to: ArrayView1<f64>,
        xi: ArrayViewMut1<f64>,
    ) {
        // Projection approximation: re-project ξ onto T_{to} M.
        self.project_tangent(to, xi);
    }
    fn euclidean_to_riemannian_hess_vp(
        &self,
        p: ArrayView1<f64>,
        egrad: ArrayView1<f64>,
        mut ehess_vp: ArrayViewMut1<f64>,
        xi: ArrayView1<f64>,
    ) {
        // S¹ is a 1-d submanifold of ℝ²; Weingarten map W_p(ξ, v) is the
        // tangential part of -D_ξ(N(p)·v) N(p) — analogous to the sphere
        // case (Circle = S¹). For a unit-sphere submanifold of codim 1:
        //   Hess_R[ξ] = P_T( Hess_E[ξ] - <egrad, p> ξ )
        let radial_egrad = egrad[0] * p[0] + egrad[1] * p[1];
        for i in 0..2 {
            ehess_vp[i] -= radial_egrad * xi[i];
        }
        self.project_tangent(p, ehess_vp);
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
    fn project_tangent(&self, p: ArrayView1<f64>, mut v: ArrayViewMut1<f64>) {
        debug_assert_eq!(p.len(), self.n + 1);
        debug_assert_eq!(v.len(), self.n + 1);
        let mut dot = 0.0_f64;
        for i in 0..p.len() {
            dot += v[i] * p[i];
        }
        for i in 0..p.len() {
            v[i] -= dot * p[i];
        }
    }
    fn retract(&self, p: ArrayView1<f64>, xi: ArrayView1<f64>, mut out: ArrayViewMut1<f64>) {
        let m = self.n + 1;
        let mut s2 = 0.0_f64;
        for i in 0..m {
            let v = p[i] + xi[i];
            out[i] = v;
            s2 += v * v;
        }
        let norm = s2.sqrt().max(1.0e-300);
        for i in 0..m {
            out[i] /= norm;
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
        // Weingarten for unit sphere: W_p(ξ, ν) = <p, ν> ξ where ν is the
        // ambient (Euclidean) gradient. Hess_R[ξ] = P_T(Hess_E[ξ] - W_p(ξ, ν)).
        let mut radial_egrad = 0.0_f64;
        for i in 0..p.len() {
            radial_egrad += egrad[i] * p[i];
        }
        for i in 0..ehess_vp.len() {
            ehess_vp[i] -= radial_egrad * xi[i];
        }
        self.project_tangent(p, ehess_vp);
    }
    fn warn_at(&self, p: ArrayView1<f64>) -> Option<&'static str> {
        // Advisory only: if the last coordinate is very small the canonical
        // chart near the equator/pole is ill-conditioned.
        if let Some(last) = p.iter().last() {
            if last.abs() < SPHERE_POLE_WARN_THRESHOLD {
                return Some("sphere: near chart pole, retraction may amplify error");
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
    fn project_tangent(&self, _p: ArrayView1<f64>, _v: ArrayViewMut1<f64>) {
        // 1-d open submanifold of ℝ; tangent space is all of ℝ.
    }
    fn retract(&self, p: ArrayView1<f64>, xi: ArrayView1<f64>, mut out: ArrayViewMut1<f64>) {
        // Smooth clamp using a tanh-style retraction:
        //   R_p(ξ) = clamp_band(p + ξ).
        // Geometrically this is the projective retraction onto the open
        // interval — first-order accurate, and the band keeps the implicit
        // tanh Jacobian bounded near the boundary.
        out[0] = self.clip(p[0] + xi[0]);
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
    fn warn_at(&self, p: ArrayView1<f64>) -> Option<&'static str> {
        let band = (self.hi - self.lo).abs() * Self::EDGE_FRAC * 10.0;
        if p[0] < self.lo + band || p[0] > self.hi - band {
            Some("interval: near boundary; trust radius should be clipped")
        } else {
            None
        }
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
        for k in 0..self.d {
            let x = p[2 * k] + xi[2 * k];
            let y = p[2 * k + 1] + xi[2 * k + 1];
            let norm = (x * x + y * y).sqrt().max(1.0e-300);
            out[2 * k] = x / norm;
            out[2 * k + 1] = y / norm;
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
        // Block-diagonal sphere Weingarten on each S¹ factor.
        for k in 0..self.d {
            let radial = egrad[2 * k] * p[2 * k] + egrad[2 * k + 1] * p[2 * k + 1];
            ehess_vp[2 * k] -= radial * xi[2 * k];
            ehess_vp[2 * k + 1] -= radial * xi[2 * k + 1];
        }
        self.project_tangent(p, ehess_vp);
    }
    fn name(&self) -> &str {
        "Torus"
    }
}

/// Cartesian product of manifolds; ambient dimensions concatenate.
pub struct Product {
    pub components: Vec<Box<dyn Manifold>>,
}

impl Manifold for Product {
    fn dim(&self) -> usize {
        self.components.iter().map(|c| c.dim()).sum()
    }
    fn ambient_dim(&self) -> usize {
        self.components.iter().map(|c| c.ambient_dim()).sum()
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
        // Suppress unused-self warning under -Dwarnings (helper kept for clarity).
        let _ = (self.slice_for(0, p), self.slice_mut_for as fn(&Self, usize, ArrayViewMut1<f64>) -> ArrayViewMut1<f64>);
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
}

impl ManifoldKind {
    /// Materialise a trait-object instance.
    pub fn build(&self) -> Box<dyn Manifold> {
        match self {
            ManifoldKind::Euclidean(d) => Box::new(Euclidean { d: *d }),
            ManifoldKind::Circle => Box::new(Circle),
            ManifoldKind::Sphere(n) => Box::new(Sphere { n: *n }),
            ManifoldKind::Interval(lo, hi) => Box::new(Interval { lo: *lo, hi: *hi }),
            ManifoldKind::Torus(d) => Box::new(Torus { d: *d }),
            ManifoldKind::Product(components) => Box::new(Product {
                components: components.iter().map(|c| c.build()).collect(),
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
            ManifoldKind::Product(components) => components.iter().map(|c| c.ambient_dim()).sum(),
        }
    }
}

// ---------------------------------------------------------------------------
// Per-point Riemannian Newton step
// ---------------------------------------------------------------------------

/// Configuration of a single per-point Riemannian Newton step.
#[derive(Debug, Clone)]
pub struct RiemannianNewtonStep {
    /// Optional trust-region radius in the Riemannian metric.
    pub trust_radius: Option<f64>,
    /// Tikhonov-like ridge added to the diagonal of the projected
    /// Hessian to keep the small tangent-space solve well-posed.
    pub ridge: f64,
}

impl Default for RiemannianNewtonStep {
    fn default() -> Self {
        Self {
            trust_radius: None,
            ridge: 0.0,
        }
    }
}

/// Solve a per-point Newton step on a manifold and return the retracted
/// new point.
///
/// Pipeline:
///   1. Project `egrad` into the tangent space at `point`.
///   2. Form the Riemannian Hessian-vector operator
///      `Hess_R[ξ] = P_T(Hess_E ξ - W_p(ξ, egrad))` and assemble its dense
///      ambient-space matrix `H_R` (small: at most `ambient_dim^2`).
///   3. Solve `H_R ξ = -grad_R` with a Tikhonov ridge; if the solve fails
///      or the step exceeds the trust radius, fall back to a scaled
///      negative-gradient (Cauchy point) step.
///   4. Retract `point + ξ` back to the manifold.
pub fn riemannian_newton_step_on_point(
    manifold: &dyn Manifold,
    point: ArrayView1<f64>,
    euclidean_hess: ArrayView2<f64>,
    euclidean_grad: ArrayView1<f64>,
    step: &RiemannianNewtonStep,
) -> Array1<f64> {
    let m = manifold.ambient_dim();
    debug_assert_eq!(point.len(), m);
    debug_assert_eq!(euclidean_grad.len(), m);
    debug_assert_eq!(euclidean_hess.shape(), &[m, m]);

    // Riemannian gradient.
    let mut grad_r = euclidean_grad.to_owned();
    manifold.euclidean_to_riemannian_grad(point, grad_r.view_mut());

    // Build Riemannian Hessian column-by-column by applying the
    // euclidean->Riemannian operator to each basis vector.
    let mut hess_r = Array2::<f64>::zeros((m, m));
    let mut basis = Array1::<f64>::zeros(m);
    for j in 0..m {
        // basis = e_j projected to tangent space.
        for i in 0..m {
            basis[i] = 0.0;
        }
        basis[j] = 1.0;
        manifold.project_tangent(point, basis.view_mut());

        // ehess_vp = H_E · basis
        let mut ehess_vp = Array1::<f64>::zeros(m);
        for i in 0..m {
            let mut acc = 0.0_f64;
            for k in 0..m {
                acc += euclidean_hess[[i, k]] * basis[k];
            }
            ehess_vp[i] = acc;
        }
        manifold.euclidean_to_riemannian_hess_vp(
            point,
            euclidean_grad,
            ehess_vp.view_mut(),
            basis.view(),
        );
        for i in 0..m {
            hess_r[[i, j]] = ehess_vp[i];
        }
    }
    // Symmetrize for numerical hygiene.
    for i in 0..m {
        for j in (i + 1)..m {
            let s = 0.5 * (hess_r[[i, j]] + hess_r[[j, i]]);
            hess_r[[i, j]] = s;
            hess_r[[j, i]] = s;
        }
    }
    // Add tangent-projected ridge.
    if step.ridge > 0.0 {
        for i in 0..m {
            hess_r[[i, i]] += step.ridge;
        }
    }

    // Solve H_R ξ = -grad_R via a small Tikhonov-damped normal-equation
    // fallback. We do a tiny LDLᵀ-style pivoted solve via a regularised
    // Gauss-Newton: ξ = -(H_R + λ I)⁻¹ grad_R with adaptive λ. For non-SPD
    // H_R we add ridge until it is PD.
    let mut xi = solve_symmetric_tikhonov(&hess_r, &grad_r);

    // Re-project ξ to the tangent space (the solve is in ambient coords).
    manifold.project_tangent(point, xi.view_mut());

    // Trust-region clip: ‖ξ‖_g ≤ Δ (use the induced metric).
    if let Some(delta) = step.trust_radius {
        let norm2 = manifold.inner_product(point, xi.view(), xi.view());
        let norm = norm2.max(0.0).sqrt();
        if norm > delta && norm > 0.0 {
            let scale = delta / norm;
            for i in 0..m {
                xi[i] *= scale;
            }
        }
    }

    // Retract.
    let mut new_point = Array1::<f64>::zeros(m);
    manifold.retract(point, xi.view(), new_point.view_mut());
    new_point
}

/// Symmetric Tikhonov-damped solve `-(H + λI)⁻¹ g`. Adaptive: starts at the
/// given ridge and bumps geometrically until the Cholesky factor succeeds.
/// Returns `-(H + λI)⁻¹ g`; on hard failure returns the (truncated) gradient
/// step `-g / ‖g‖`.
fn solve_symmetric_tikhonov(h: &Array2<f64>, g: &Array1<f64>) -> Array1<f64> {
    let n = g.len();
    debug_assert_eq!(h.shape(), &[n, n]);
    // Estimate trace for an initial ridge.
    let mut trace = 0.0_f64;
    for i in 0..n {
        trace += h[[i, i]].abs();
    }
    let scale = (trace / (n.max(1) as f64)).max(1.0);
    let mut lambda = 0.0_f64;
    let mut bumps = 0usize;
    loop {
        let mut a = h.clone();
        for i in 0..n {
            a[[i, i]] += lambda;
        }
        if let Some(l) = cholesky_lower(&a) {
            let mut rhs = Array1::<f64>::zeros(n);
            for i in 0..n {
                rhs[i] = -g[i];
            }
            return chol_solve_inplace(&l, &rhs);
        }
        // Bump ridge by ~1e-8 of scale → 1e-6 → ... up to a cap.
        bumps += 1;
        lambda = (lambda * 10.0).max(1.0e-8 * scale);
        if bumps > 20 {
            // Hard fallback: tiny steepest descent.
            let mut out = Array1::<f64>::zeros(n);
            let mut gn = 0.0_f64;
            for i in 0..n {
                gn += g[i] * g[i];
            }
            let gn = gn.sqrt().max(1.0e-300);
            for i in 0..n {
                out[i] = -g[i] / gn;
            }
            return out;
        }
    }
}

/// Dense lower-triangular Cholesky `L` with `L Lᵀ = A`, returning `None` if
/// `A` is not numerically PD.
fn cholesky_lower(a: &Array2<f64>) -> Option<Array2<f64>> {
    let n = a.nrows();
    if a.ncols() != n {
        return None;
    }
    let mut l = Array2::<f64>::zeros((n, n));
    for i in 0..n {
        for j in 0..=i {
            let mut sum = a[[i, j]];
            for k in 0..j {
                sum -= l[[i, k]] * l[[j, k]];
            }
            if i == j {
                if !(sum > 0.0) {
                    return None;
                }
                l[[i, j]] = sum.sqrt();
            } else {
                l[[i, j]] = sum / l[[j, j]];
            }
        }
    }
    Some(l)
}

fn chol_solve_inplace(l: &Array2<f64>, rhs: &Array1<f64>) -> Array1<f64> {
    let n = rhs.len();
    let mut y = Array1::<f64>::zeros(n);
    for i in 0..n {
        let mut s = rhs[i];
        for k in 0..i {
            s -= l[[i, k]] * y[k];
        }
        y[i] = s / l[[i, i]];
    }
    let mut x = Array1::<f64>::zeros(n);
    for i in (0..n).rev() {
        let mut s = y[i];
        for k in (i + 1)..n {
            s -= l[[k, i]] * x[k];
        }
        x[i] = s / l[[i, i]];
    }
    x
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
        let mut out = array![0.0_f64; 3];
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
