use std::fmt;

use ndarray::{Array1, Array2, ArrayView1, ArrayView2};

pub const GEOMETRY_EPS: f64 = 1.0e-12;

#[derive(Debug, Clone, PartialEq)]
pub enum GeometryError {
    DimensionMismatch {
        context: &'static str,
        expected: usize,
        got: usize,
    },
    InvalidPoint(&'static str),
    Singular(&'static str),
    /// A manifold primitive has no implementation for this manifold and must
    /// not silently fall back to a wrong default (e.g. a curved-manifold VJP
    /// for which no closed form is wired up yet).
    Unsupported(&'static str),
}

impl fmt::Display for GeometryError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::DimensionMismatch {
                context,
                expected,
                got,
            } => write!(f, "{context} expected length {expected}, got {got}"),
            Self::InvalidPoint(message) => write!(f, "invalid manifold point: {message}"),
            Self::Singular(message) => write!(f, "singular geometry operation: {message}"),
            Self::Unsupported(message) => write!(f, "unsupported geometry operation: {message}"),
        }
    }
}

impl std::error::Error for GeometryError {}

pub type GeometryResult<T> = Result<T, GeometryError>;

pub trait RiemannianManifold: Send + Sync {
    fn dim(&self) -> usize;

    fn ambient_dim(&self) -> usize {
        self.dim()
    }

    fn tangent_basis(&self, point: ArrayView1<'_, f64>) -> GeometryResult<Array2<f64>>;

    fn exp_map(
        &self,
        point: ArrayView1<'_, f64>,
        tangent_vec: ArrayView1<'_, f64>,
    ) -> GeometryResult<Array1<f64>>;

    fn log_map(
        &self,
        p_from: ArrayView1<'_, f64>,
        p_to: ArrayView1<'_, f64>,
    ) -> GeometryResult<Array1<f64>>;

    fn parallel_transport(
        &self,
        point_along: ArrayView2<'_, f64>,
        vec: ArrayView1<'_, f64>,
    ) -> GeometryResult<Array1<f64>>;

    fn metric_tensor(&self, point: ArrayView1<'_, f64>) -> GeometryResult<Array2<f64>>;

    fn christoffel_symbols(&self, point: ArrayView1<'_, f64>) -> GeometryResult<Vec<Array2<f64>>> {
        check_len("Christoffel point", point.len(), self.ambient_dim())?;
        Err(GeometryError::Unsupported(
            "Christoffel symbols require a manifold-specific local chart",
        ))
    }

    fn sectional_curvature(
        &self,
        point: ArrayView1<'_, f64>,
        tangent_pair: (ArrayView1<'_, f64>, ArrayView1<'_, f64>),
    ) -> GeometryResult<f64>;

    fn project_tangent(
        &self,
        point: ArrayView1<'_, f64>,
        vec: ArrayView1<'_, f64>,
    ) -> GeometryResult<Array1<f64>> {
        // Default projection is the identity (Euclidean-flat tangent space).
        // Validate that BOTH the base point and the tangent vector live in the
        // ambient space so a caller passing a wrong-length vector fails fast
        // here rather than producing a silently mis-shaped tangent vector. The
        // tangent of `T_pM` is represented in the same ambient coordinates as
        // the point, so its length must equal `ambient_dim()` too.
        let expected = self.ambient_dim();
        if point.len() != expected {
            return Err(GeometryError::DimensionMismatch {
                context: "project_tangent point",
                expected,
                got: point.len(),
            });
        }
        if vec.len() != expected {
            return Err(GeometryError::DimensionMismatch {
                context: "project_tangent vector",
                expected,
                got: vec.len(),
            });
        }
        Ok(vec.to_owned())
    }

    /// Riemannian gradient of a scalar `f` raised from its **ambient Euclidean
    /// differential** `e` — the vector `∂f/∂x` in ambient coordinates that an
    /// objective returns from its `value_gradient`.
    ///
    /// The Riemannian gradient is the Riesz representative of the differential
    /// under the manifold metric `g`: the unique tangent vector `v` satisfying
    ///
    /// ```text
    ///   g_x(v, ξ) = Df_x[ξ] = ⟨e, ξ⟩   for every tangent ξ.
    /// ```
    ///
    /// Orthogonally projecting `e` onto the tangent space ([`project_tangent`])
    /// produces `v` **only** for the embedded/identity metric. For a genuine
    /// Riemannian metric (affine-invariant SPD, canonical Stiefel, …) the
    /// differential must be *raised through the metric* — projecting alone gives
    /// the wrong direction and the wrong slope, so any model linear term or
    /// Armijo slope built from it is not even first-order accurate (issue #955).
    ///
    /// The default raises `e` in a tangent basis `B = tangent_basis(x)` against
    /// the metric `G = metric_tensor(x)`:
    ///
    /// ```text
    ///   v = B (Bᵀ G B)⁻¹ Bᵀ e.
    /// ```
    ///
    /// This is the Riesz representative for ANY basis `B` of `T_xM` (proof: for
    /// `ξ = B c`, `g_x(v, ξ) = eᵀ B (Bᵀ G B)⁻¹ (Bᵀ G B) c = eᵀ B c = ⟨e, ξ⟩`),
    /// and it collapses to the orthogonal tangent projection `B Bᵀ e` exactly
    /// when `B` is metric-orthonormal / the metric is the embedded one. It is the
    /// mathematically correct fallback, so a future non-identity-metric manifold
    /// is never silently first-order wrong.
    ///
    /// Manifolds whose tangent projection already coincides with this (every
    /// *embedded* manifold carrying the induced metric — Euclidean, Sphere,
    /// Circle, Torus, Grassmann) override with the O(m) `project_tangent`;
    /// manifolds with a slick closed form (SPD: `P·sym(E)·P`; Stiefel:
    /// `E − Y Eᵀ Y`) override with that, avoiding the dense `m×m` metric tensor.
    fn riemannian_gradient(
        &self,
        point: ArrayView1<'_, f64>,
        euclidean_grad: ArrayView1<'_, f64>,
    ) -> GeometryResult<Array1<f64>> {
        let m = self.ambient_dim();
        check_len("riemannian_gradient point", point.len(), m)?;
        check_len(
            "riemannian_gradient euclidean_grad",
            euclidean_grad.len(),
            m,
        )?;
        let b = self.tangent_basis(point)?; // m × d
        let g = self.metric_tensor(point)?; // m × m
        // Bᵀ e  (length d) and the Gram matrix Bᵀ G B  (d × d).
        let bt = b.t();
        let bte = bt.dot(&euclidean_grad.to_owned());
        let gb = g.dot(&b);
        let btgb = bt.dot(&gb);
        if btgb.nrows() == 0 {
            // A zero-dimensional tangent space (no degrees of freedom): the only
            // tangent vector is 0.
            return Ok(Array1::<f64>::zeros(m));
        }
        // Solve (BᵀGB) c = Bᵀ e for the basis coordinates of v, then v = B c.
        let c = inverse(&btgb)?.dot(&bte);
        Ok(b.dot(&c))
    }

    fn retract(
        &self,
        point: ArrayView1<'_, f64>,
        tangent_vec: ArrayView1<'_, f64>,
    ) -> GeometryResult<Array1<f64>> {
        self.exp_map(point, tangent_vec)
    }

    /// Whether [`retract`](Self::retract) is at least a SECOND-ORDER retraction,
    /// i.e. `D²(f∘R_x)(0) = Hess f(x)` for all `f`, so the trust-region quadratic
    /// model built from the Riemannian Hessian is a valid second-order model of
    /// `f` along the retraction (issue #956).
    ///
    /// Manifolds whose `retract` is the exponential map or another second-order
    /// retraction return `true` (the default — the default `retract` *is*
    /// `exp_map`, which is second-order). A manifold exposing only a FIRST-ORDER
    /// retraction (e.g. the Stiefel/Grassmann QR retraction `qf(Y + Δ)`, whose
    /// acceleration at `0` is not normal to the manifold) must override this to
    /// `false`: the linear model term `Df_x[η]` is retraction-independent and
    /// stays correct, but the Riemannian-Hessian quadratic term is *not* the
    /// second derivative of `f∘R_x` and would corrupt the predicted-vs-actual
    /// reduction ratio `ρ` and hence the trust-region radius control. The trust
    /// region falls back to the first-order-correct Cauchy model in that case.
    fn retraction_is_second_order(&self) -> bool {
        true
    }

    /// Vector–Jacobian product of the ambient map `exp_p(v)`.
    ///
    /// Given a cotangent `grad_output` w.r.t. the ambient output of
    /// [`exp_map`](Self::exp_map), return `(grad_point, grad_tangent)`, the
    /// pullbacks w.r.t. the base point `p` and the (raw, unprojected) tangent
    /// input `v`. This is the analytic backward used by reverse-mode autodiff
    /// wrappers (e.g. the Python `torch.autograd.Function` around
    /// `manifold_exp_map`); it must never be the silent straight-through
    /// identity for a curved manifold.
    ///
    /// The default is the exact VJP for *flat* manifolds, where
    /// `exp_p(v) = p + v` in ambient coordinates and so both Jacobians are the
    /// identity (Euclidean, Circle, Torus, and products thereof). Curved
    /// manifolds **must** override this with their analytic Jacobi-field VJP;
    /// a manifold without a closed form must override it to return an error
    /// rather than inherit the wrong identity default.
    fn exp_map_vjp(
        &self,
        point: ArrayView1<'_, f64>,
        tangent_vec: ArrayView1<'_, f64>,
        grad_output: ArrayView1<'_, f64>,
    ) -> GeometryResult<(Array1<f64>, Array1<f64>)> {
        let m = self.ambient_dim();
        check_len("exp_map_vjp point", point.len(), m)?;
        check_len("exp_map_vjp tangent", tangent_vec.len(), m)?;
        check_len("exp_map_vjp grad_output", grad_output.len(), m)?;
        Ok((grad_output.to_owned(), grad_output.to_owned()))
    }
}

#[derive(Debug, Clone, PartialEq)]
pub enum ManifoldSpec {
    Euclidean(usize),
    Circle,
    Sphere { intrinsic_dim: usize },
    Torus { dim: usize },
    Grassmann { k: usize, n: usize },
    Stiefel { k: usize, n: usize },
    Spd { n: usize },
    Product(Vec<ManifoldSpec>),
}

impl ManifoldSpec {
    /// Instantiate the concrete [`RiemannianManifold`] for this descriptor.
    ///
    /// Fallible because the constrained-frame families have nonempty domains:
    /// `Gr(k, n)` and `St(n, k)` exist only for `1 ≤ k ≤ n`. An out-of-domain
    /// descriptor is rejected here (and recursively for [`Product`] parts)
    /// before any dimension, projection, exponential, or curvature computation
    /// can run on a nonexistent manifold.
    ///
    /// [`Product`]: Self::Product
    pub fn build(&self) -> GeometryResult<Box<dyn RiemannianManifold>> {
        match self {
            Self::Euclidean(dim) => Ok(Box::new(crate::geometry::EuclideanManifold::new(*dim))),
            Self::Circle => Ok(Box::new(crate::geometry::CircleManifold::new())),
            Self::Sphere { intrinsic_dim } => Ok(Box::new(crate::geometry::SphereManifold::new(
                *intrinsic_dim,
            ))),
            Self::Torus { dim } => Ok(Box::new(crate::geometry::TorusManifold::new(*dim))),
            Self::Grassmann { k, n } => {
                Ok(Box::new(crate::geometry::GrassmannManifold::new(*k, *n)?))
            }
            Self::Stiefel { k, n } => Ok(Box::new(crate::geometry::StiefelManifold::new(*k, *n)?)),
            Self::Spd { n } => Ok(Box::new(crate::geometry::SpdManifold::new(*n))),
            Self::Product(parts) => {
                let mut built = Vec::with_capacity(parts.len());
                for part in parts {
                    built.push(part.build()?);
                }
                Ok(Box::new(crate::geometry::ProductManifold::new(built)))
            }
        }
    }
}

pub(crate) const fn check_len(
    context: &'static str,
    got: usize,
    expected: usize,
) -> GeometryResult<()> {
    if got == expected {
        Ok(())
    } else {
        Err(GeometryError::DimensionMismatch {
            context,
            expected,
            got,
        })
    }
}

pub(crate) fn dot(a: ArrayView1<'_, f64>, b: ArrayView1<'_, f64>) -> f64 {
    assert_eq!(a.len(), b.len());
    let mut out = 0.0;
    for i in 0..a.len() {
        out += a[i] * b[i];
    }
    out
}

/// Multi-GPU row-tiled matrix product `A·B`, fanned across **all** usable
/// devices.
///
/// `A` is `m×k` and `B` is `k×n`; the result is `m×n`. The single-device
/// `fast_ab` shim already offloads this GEMM, but it pins the launch to the
/// primary device. For a tall `A` (many independent output rows — the common
/// case when a manifold operation is applied to a large batch of points/atoms),
/// the rows split cleanly across the pool: we reshape `A` into a
/// `tiles × rows_per_tile × k` batch and call the broadcast-`B` strided-batched
/// GEMM, which [`crate::gpu::pool::scatter_batched`]es one cuBLAS call per device
/// on its own bound context (`b` is shared across every tile). The output tiles
/// are stitched back into the `m×n` result. Any leftover rows that don't fill a
/// whole tile, and the entire batch when the pool has one device / the workload
/// is below the multi-GPU floor / the runtime is unavailable, fall through to the
/// auto-dispatch `fast_ab` (single-device GPU or faer). f64 throughout, so the
/// result is identical regardless of which path produced it.
///
/// Choosing the tiling: we target as many equal tiles as there are output rows
/// can support while keeping each tile a non-trivial GEMM, so the batch axis is
/// long enough to cross `crate::gpu::linalg_dispatch`'s multi-GPU batch floor and spread
/// across every device.
pub(crate) fn fast_ab_rows_multi_gpu(
    a: ArrayView2<'_, f64>,
    b: ArrayView2<'_, f64>,
) -> Array2<f64> {
    use crate::linalg::faer_ndarray::fast_ab;
    let (m, k) = a.dim();
    let (kb, n) = b.dim();
    assert_eq!(k, kb, "fast_ab_rows_multi_gpu inner dimension mismatch");

    // Only worth the reshape/stitch overhead when the pool actually has more than
    // one device and there are enough rows to tile across it; otherwise the plain
    // single-device shim is strictly better.
    let multi_gpu =
        crate::gpu::device_runtime::GpuRuntime::global().is_some_and(|rt| rt.device_count() > 1);
    // The batch axis must clear the multi-GPU floor used inside the dispatch
    // layer (64) for the split to engage, so we need at least that many tiles.
    const MIN_TILES: usize = 64;
    const MIN_TILE_ROWS: usize = 4;
    if multi_gpu && m >= MIN_TILES * MIN_TILE_ROWS && n > 0 {
        let rows_per_tile = (m / MIN_TILES).max(MIN_TILE_ROWS);
        let tiles = m / rows_per_tile;
        let covered = tiles * rows_per_tile;
        // Reshape the first `covered` rows into a tiles×rows_per_tile×k batch
        // (row-major reshape is exactly the row-block tiling we want).
        let a3 = a
            .slice(ndarray::s![0..covered, ..])
            .to_owned()
            .into_shape_with_order((tiles, rows_per_tile, k));
        if let Ok(a3) = a3 {
            if let Some(result3) = crate::gpu::try_fast_ab_broadcast_b_batched(a3.view(), b.view())
            {
                let mut out = Array2::<f64>::zeros((m, n));
                for t in 0..tiles {
                    let block = result3.index_axis(ndarray::Axis(0), t);
                    out.slice_mut(ndarray::s![t * rows_per_tile..(t + 1) * rows_per_tile, ..])
                        .assign(&block);
                }
                // Tail rows that didn't fill a whole tile finish on the
                // single-device shim; the result is bit-identical f64.
                if covered < m {
                    let tail = fast_ab(&a.slice(ndarray::s![covered..m, ..]), &b);
                    out.slice_mut(ndarray::s![covered..m, ..]).assign(&tail);
                }
                return out;
            }
        }
    }
    // Single device / small batch / no runtime: plain auto-dispatch GEMM.
    fast_ab(&a, &b)
}

pub(crate) fn norm(a: ArrayView1<'_, f64>) -> f64 {
    dot(a, a).sqrt()
}

/// Metric inner product `aᵀ G b` for a (symmetric) metric tensor `G`.
///
/// For a manifold whose `metric_tensor` is the ambient identity this reduces
/// to the Euclidean `dot`; for one with a genuine Riemannian metric (e.g. the
/// affine-invariant SPD metric) it evaluates the correct geometric inner
/// product on the tangent space.
pub(crate) fn quad_form(
    g: ArrayView2<'_, f64>,
    a: ArrayView1<'_, f64>,
    b: ArrayView1<'_, f64>,
) -> f64 {
    let n = a.len();
    assert_eq!(g.nrows(), n);
    assert_eq!(g.ncols(), b.len());
    // aᵀ G b: the inner matrix–vector product G·b is the O(n²) cost and is the
    // hot kernel of every metric inner product (g_inner / g_norm) and of the
    // metric Gram–Schmidt tangent basis. Route it through the GPU-dispatched
    // fast_av shim so large-ambient metrics (SPD/Stiefel/Grassmann n²×n²) offload
    // to the GPU; the trailing a·(Gb) is an O(n) dot.
    let gb = crate::linalg::faer_ndarray::fast_av(&g, &b);
    dot(a, gb.view())
}

pub(crate) fn identity(n: usize) -> Array2<f64> {
    let mut out = Array2::<f64>::zeros((n, n));
    for i in 0..n {
        out[[i, i]] = 1.0;
    }
    out
}

pub(crate) fn zero_christoffel(dim: usize) -> Vec<Array2<f64>> {
    (0..dim).map(|_| Array2::<f64>::zeros((dim, dim))).collect()
}

pub(crate) fn wrap_angle(theta: f64) -> f64 {
    let two_pi = std::f64::consts::PI * 2.0;
    (theta + std::f64::consts::PI).rem_euclid(two_pi) - std::f64::consts::PI
}

pub(crate) fn sym(a: &Array2<f64>) -> Array2<f64> {
    let mut out = a.clone();
    for i in 0..a.nrows() {
        for j in 0..a.ncols() {
            out[[i, j]] = 0.5 * (a[[i, j]] + a[[j, i]]);
        }
    }
    out
}

pub(crate) fn from_flat(
    v: ArrayView1<'_, f64>,
    rows: usize,
    cols: usize,
) -> GeometryResult<Array2<f64>> {
    check_len("flat matrix", v.len(), rows * cols)?;
    let mut out = Array2::<f64>::zeros((rows, cols));
    for i in 0..rows {
        for j in 0..cols {
            out[[i, j]] = v[i * cols + j];
        }
    }
    Ok(out)
}

pub(crate) fn flatten(a: &Array2<f64>) -> Array1<f64> {
    let mut out = Array1::<f64>::zeros(a.nrows() * a.ncols());
    for i in 0..a.nrows() {
        for j in 0..a.ncols() {
            out[i * a.ncols() + j] = a[[i, j]];
        }
    }
    out
}

/// Build a **Euclidean-orthonormal** basis of the tangent space at `point` by
/// modified Gram–Schmidt over the projected ambient standard basis.
///
/// The returned columns satisfy `Qᵀ Q = I` under the *ambient Euclidean* inner
/// product (the plain `dot`). This is the correct, intended basis for a
/// manifold whose Riemannian metric *is* the embedded Euclidean metric on its
/// horizontal tangent space — notably the **Grassmann** manifold, where the
/// tangent inner product is `tr(Δ₁ᵀΔ₂)`.
///
/// It is **not** metric-orthonormal for a manifold with a non-Euclidean metric
/// (Stiefel's canonical metric `⟨Δ₁,Δ₂⟩ = tr(Δ₁ᵀ(I−½YYᵀ)Δ₂)`, or SPD's
/// affine-invariant metric): for those, use
/// [`tangent_basis_metric_orthonormal`], which Gram–Schmidts under the
/// manifold's own `metric_tensor`.
///
/// This is the shared engine behind [`tangent_basis`](RiemannianManifold::tangent_basis)
/// for the matrix manifolds whose tangent space has no closed-form basis. It
/// walks the `n × k` standard basis in column-major order (outer `col`, inner
/// `row`), projects each `e_{row,col}` onto the tangent space via
/// `m.project_tangent`, re-orthogonalizes against the columns accepted so far,
/// and keeps it iff its residual norm exceeds the `1e-10` drop tolerance,
/// stopping the moment `m.dim()` independent directions have been collected.
/// Each caller keeps its own input validation and then delegates here, so the
/// numerically delicate orthogonalization order, drop tolerance, and early-exit
/// logic live in exactly one place.
pub(crate) fn projected_standard_basis_tangent<M: RiemannianManifold + ?Sized>(
    m: &M,
    point: ArrayView1<'_, f64>,
    n: usize,
    k: usize,
) -> GeometryResult<Array2<f64>> {
    let mut columns: Vec<Array1<f64>> = Vec::with_capacity(m.dim());
    for col in 0..k {
        for row in 0..n {
            let mut e = Array2::<f64>::zeros((n, k));
            e[[row, col]] = 1.0;
            let mut v = m.project_tangent(point, flatten(&e).view())?;
            for q in &columns {
                let proj = dot(q.view(), v.view());
                v -= &(q * proj);
            }
            let nrm = dot(v.view(), v.view()).sqrt();
            if nrm > 1.0e-10 {
                columns.push(v / nrm);
            }
            if columns.len() == m.dim() {
                let mut out = Array2::<f64>::zeros((m.ambient_dim(), m.dim()));
                for j in 0..columns.len() {
                    for i in 0..m.ambient_dim() {
                        out[[i, j]] = columns[j][i];
                    }
                }
                return Ok(out);
            }
        }
    }
    Ok(Array2::<f64>::zeros((m.ambient_dim(), columns.len())))
}

/// Build a **metric-orthonormal** basis of the tangent space at `point`, i.e. a
/// set of columns `Q` satisfying `Qᵀ W Q = I` where `W = m.metric_tensor(point)`
/// is the manifold's Riemannian metric in flattened ambient coordinates.
///
/// This is the correct tangent basis for a manifold whose metric is **not** the
/// embedded Euclidean inner product — Stiefel's canonical metric
/// `⟨Δ₁,Δ₂⟩ = tr(Δ₁ᵀ(I−½YYᵀ)Δ₂)` and SPD's affine-invariant metric. (For a
/// Euclidean-metric manifold like Grassmann, `W = I` and this coincides with
/// [`projected_standard_basis_tangent`].)
///
/// Same projected-standard-basis walk as the Euclidean routine, but every inner
/// product is the metric inner product `⟨u,v⟩_W = uᵀ W v` (via
/// [`quad_form`]): Gram–Schmidt projections subtract `⟨q,v⟩_W · q` and the
/// retained columns are normalized by `‖v‖_W = sqrt(⟨v,v⟩_W)`, so the resulting
/// `Q` is orthonormal *in the manifold's metric*.
///
/// Concretely on `St(3, 2)` at `Y = [e₁, e₂]`, the vertical tangent
/// `Δ = Y·[[0,−1],[1,0]]` has Euclidean norm² 2 but canonical-metric norm² 1, so
/// a metric-orthonormal basis must reflect that — the Euclidean routine would
/// mis-scale it.
pub(crate) fn tangent_basis_metric_orthonormal<M: RiemannianManifold + ?Sized>(
    m: &M,
    point: ArrayView1<'_, f64>,
    n: usize,
    k: usize,
) -> GeometryResult<Array2<f64>> {
    let w = m.metric_tensor(point)?;
    let mut columns: Vec<Array1<f64>> = Vec::with_capacity(m.dim());
    for col in 0..k {
        for row in 0..n {
            let mut e = Array2::<f64>::zeros((n, k));
            e[[row, col]] = 1.0;
            let mut v = m.project_tangent(point, flatten(&e).view())?;
            for q in &columns {
                let proj = quad_form(w.view(), q.view(), v.view());
                v -= &(q * proj);
            }
            let nrm = quad_form(w.view(), v.view(), v.view()).max(0.0).sqrt();
            if nrm > 1.0e-10 {
                columns.push(v / nrm);
            }
            if columns.len() == m.dim() {
                let mut out = Array2::<f64>::zeros((m.ambient_dim(), m.dim()));
                for j in 0..columns.len() {
                    for i in 0..m.ambient_dim() {
                        out[[i, j]] = columns[j][i];
                    }
                }
                return Ok(out);
            }
        }
    }
    Ok(Array2::<f64>::zeros((m.ambient_dim(), columns.len())))
}

/// Thin/compact Gram–Schmidt QR factorization `A = Q·R` for an `n×k` input
/// (`n ≥ k`). The returned `Q` is `n×k` with **orthonormal columns**
/// (`QᵀQ = I`) and `R` is `k×k` upper-triangular.
///
/// On a rank-deficient column (residual ≈ 0 after orthogonalizing against the
/// previously accepted columns) the diagonal `R[j, j]` is set to 0 and a
/// *fallback* unit column is synthesized so the column count stays `k` and `Q`
/// remains a valid orthonormal frame. The fallback is a standard axis `e_a`
/// Gram–Schmidted against ALL previously accepted columns and renormalized; if
/// that residual also vanishes (the axis lies in the accepted span) the next
/// axis is tried, until an axis with a nonzero orthogonal residual is found.
/// Simply planting `e_j` (the old behavior) breaks orthonormality — e.g. two
/// identical columns `(1,1)/√2` would yield a fallback `e₂` with
/// `q₁·q₂ = 1/√2 ≠ 0`.
pub(crate) fn qr_thin(a: &Array2<f64>) -> (Array2<f64>, Array2<f64>) {
    let n = a.nrows();
    let k = a.ncols();
    let mut q = Array2::<f64>::zeros((n, k));
    let mut r = Array2::<f64>::zeros((k, k));
    for j in 0..k {
        let mut v = a.column(j).to_owned();
        for i in 0..j {
            let qi = q.column(i);
            let rij = dot(qi, v.view());
            r[[i, j]] = rij;
            for row in 0..n {
                v[row] -= rij * q[[row, i]];
            }
        }
        let nrm = norm(v.view());
        if nrm > GEOMETRY_EPS {
            r[[j, j]] = nrm;
            for row in 0..n {
                q[[row, j]] = v[row] / nrm;
            }
        } else {
            // Rank-deficient column: `R[j, j] = 0`. Synthesize a fallback unit
            // column orthogonal to ALL accepted columns 0..j by Gram–Schmidting
            // a standard axis against them; try successive axes until one has a
            // nonzero orthogonal residual (always succeeds for j < n since the
            // accepted columns span a j-dimensional subspace of ℝⁿ, leaving an
            // (n−j)-dimensional orthogonal complement that at least one axis
            // touches).
            r[[j, j]] = 0.0;
            for axis in 0..n {
                let mut f = Array1::<f64>::zeros(n);
                f[axis] = 1.0;
                for i in 0..j {
                    let qi = q.column(i);
                    let proj = dot(qi, f.view());
                    for row in 0..n {
                        f[row] -= proj * q[[row, i]];
                    }
                }
                let fnrm = norm(f.view());
                if fnrm > GEOMETRY_EPS {
                    for row in 0..n {
                        q[[row, j]] = f[row] / fnrm;
                    }
                    break;
                }
            }
        }
    }
    (q, r)
}

pub(crate) fn inverse(a: &Array2<f64>) -> GeometryResult<Array2<f64>> {
    let n = a.nrows();
    if n != a.ncols() {
        return Err(GeometryError::Singular("inverse requires a square matrix"));
    }
    let mut aug = Array2::<f64>::zeros((n, 2 * n));
    for i in 0..n {
        for j in 0..n {
            aug[[i, j]] = a[[i, j]];
        }
        aug[[i, n + i]] = 1.0;
    }
    for col in 0..n {
        let mut pivot = col;
        let mut best = aug[[col, col]].abs();
        for row in col + 1..n {
            let val = aug[[row, col]].abs();
            if val > best {
                best = val;
                pivot = row;
            }
        }
        if best < GEOMETRY_EPS {
            return Err(GeometryError::Singular("matrix inverse pivot underflow"));
        }
        if pivot != col {
            for j in 0..2 * n {
                let tmp = aug[[col, j]];
                aug[[col, j]] = aug[[pivot, j]];
                aug[[pivot, j]] = tmp;
            }
        }
        let scale = aug[[col, col]];
        for j in 0..2 * n {
            aug[[col, j]] /= scale;
        }
        for row in 0..n {
            if row == col {
                continue;
            }
            let factor = aug[[row, col]];
            for j in 0..2 * n {
                aug[[row, j]] -= factor * aug[[col, j]];
            }
        }
    }
    let mut out = Array2::<f64>::zeros((n, n));
    for i in 0..n {
        for j in 0..n {
            out[[i, j]] = aug[[i, n + j]];
        }
    }
    Ok(out)
}

/// Sweep budget multiplier for the classical Jacobi eigensolver: the iteration
/// cap is `JACOBI_SWEEP_BUDGET · n²`. Classical (largest-off-diagonal) Jacobi
/// converges quadratically once the off-diagonals are small, needing only a
/// handful of full `O(n²)` sweeps; this generous multiple lets even clustered
/// spectra finish while still failing loudly on a genuinely stalled matrix.
const JACOBI_SWEEP_BUDGET: usize = 64;

/// Relative off-diagonal convergence threshold for [`jacobi_symmetric`]: the
/// largest off-diagonal magnitude must fall below `JACOBI_REL_TOL · ‖A‖_F`. Near
/// `f64` precision so the diagonalization is accurate to working precision.
const JACOBI_REL_TOL: f64 = 1.0e-13;

pub(crate) fn jacobi_symmetric(a: &Array2<f64>) -> GeometryResult<(Array1<f64>, Array2<f64>)> {
    let n = a.nrows();
    if n != a.ncols() {
        return Err(GeometryError::InvalidPoint(
            "Jacobi eigensolver requires square input",
        ));
    }
    let mut d = sym(a);
    let mut v = identity(n);
    let max_iter = JACOBI_SWEEP_BUDGET * n.max(1) * n.max(1);
    // Relative convergence threshold: the largest off-diagonal magnitude must
    // fall to `1e-13 * ||A||_F`. A fixed absolute `1e-13` is meaningless for
    // matrices whose scale is far from unity (a well-scaled large-norm matrix
    // could never reach it; a tiny-norm matrix would "converge" trivially),
    // and silently returning the partially-diagonalized state after exhausting
    // `max_iter` hides genuine non-convergence (e.g. clustered/degenerate
    // spectra that stall the classical sweep). The Frobenius norm is invariant
    // under the orthogonal Jacobi rotations, so it is computed once from the
    // symmetrized input.
    let frob_norm = {
        let mut acc = 0.0;
        for i in 0..n {
            for j in 0..n {
                acc += d[[i, j]] * d[[i, j]];
            }
        }
        acc.sqrt()
    };
    let threshold = JACOBI_REL_TOL * frob_norm;
    let mut converged = false;
    for _ in 0..max_iter {
        let mut p = 0usize;
        let mut q = 0usize;
        let mut best = 0.0;
        for i in 0..n {
            for j in i + 1..n {
                let val = d[[i, j]].abs();
                if val > best {
                    best = val;
                    p = i;
                    q = j;
                }
            }
        }
        // `best <= threshold` (rather than `<`) makes the exactly-diagonal and
        // zero-norm cases (`best == threshold == 0`) converge immediately.
        if best <= threshold {
            converged = true;
            break;
        }
        let tau = (d[[q, q]] - d[[p, p]]) / (2.0 * d[[p, q]]);
        let t = tau.signum() / (tau.abs() + (1.0 + tau * tau).sqrt());
        let c = 1.0 / (1.0 + t * t).sqrt();
        let s = t * c;
        for k in 0..n {
            let dpk = d[[p, k]];
            let dqk = d[[q, k]];
            d[[p, k]] = c * dpk - s * dqk;
            d[[q, k]] = s * dpk + c * dqk;
        }
        for k in 0..n {
            let dkp = d[[k, p]];
            let dkq = d[[k, q]];
            d[[k, p]] = c * dkp - s * dkq;
            d[[k, q]] = s * dkp + c * dkq;
        }
        for k in 0..n {
            let vkp = v[[k, p]];
            let vkq = v[[k, q]];
            v[[k, p]] = c * vkp - s * vkq;
            v[[k, q]] = s * vkp + c * vkq;
        }
    }
    if !converged {
        return Err(GeometryError::Singular(
            "Jacobi eigensolver did not converge within max_iter (off-diagonal mass above 1e-13 * Frobenius norm)",
        ));
    }
    let mut evals = Array1::<f64>::zeros(n);
    for i in 0..n {
        evals[i] = d[[i, i]];
    }
    Ok((evals, v))
}

pub(crate) fn spectral_map_spd(
    a: &Array2<f64>,
    f: impl Fn(f64) -> GeometryResult<f64>,
) -> GeometryResult<Array2<f64>> {
    let (evals, evecs) = jacobi_symmetric(a)?;
    let n = a.nrows();
    let mut diag = Array2::<f64>::zeros((n, n));
    for i in 0..n {
        if evals[i] <= 0.0 || !evals[i].is_finite() {
            return Err(GeometryError::InvalidPoint(
                "SPD eigenvalue is not positive",
            ));
        }
        diag[[i, i]] = f(evals[i])?;
    }
    // Reconstruction V·f(Λ)·Vᵀ: two dense n×n products GPU-dispatched via
    // fast_ab/fast_abt for large ambient dimension.
    use crate::linalg::faer_ndarray::{fast_ab, fast_abt};
    Ok(fast_abt(&fast_ab(&evecs, &diag), &evecs))
}

pub(crate) fn spectral_map_symmetric(
    a: &Array2<f64>,
    f: impl Fn(f64) -> GeometryResult<f64>,
) -> GeometryResult<Array2<f64>> {
    let (evals, evecs) = jacobi_symmetric(a)?;
    let n = a.nrows();
    let mut diag = Array2::<f64>::zeros((n, n));
    for i in 0..n {
        diag[[i, i]] = f(evals[i])?;
    }
    // Reconstruction V·f(Λ)·Vᵀ, GPU-dispatched via fast_ab/fast_abt.
    use crate::linalg::faer_ndarray::{fast_ab, fast_abt};
    Ok(fast_abt(&fast_ab(&evecs, &diag), &evecs))
}

/// Thin singular value decomposition of a tall matrix `Y` (`n × k`, `n ≥ k`)
/// via the symmetric eigendecomposition of the small `k × k` Gram matrix
/// `YᵀY = V Σ² Vᵀ`: returns `(U, σ, V)` with `Y = U diag(σ) Vᵀ`, where `U` is
/// `n × k` with orthonormal columns spanning `range(Y)`, `σ` holds the singular
/// values, and `V` is `k × k` orthogonal. Forming the Gram keeps the
/// eigenproblem at the small dimension `k`; the two products that carry the
/// large ambient dimension `n` (`YᵀY` and `U = Y V Σ⁻¹`) are GPU-dispatched.
///
/// A numerically-zero singular value (`σ ≤ GEOMETRY_EPS`) leaves the
/// corresponding `U` column zero rather than dividing through. Callers that
/// require full rank (e.g. [`polar_factor`]) reject it, while callers for which
/// a rank-deficient input is admissible (the Grassmann/Stiefel geodesic, where
/// a zero singular value is a vanishing principal angle) keep the zero column.
pub(crate) fn thin_svd_gram(
    y: &Array2<f64>,
) -> GeometryResult<(Array2<f64>, Array1<f64>, Array2<f64>)> {
    use crate::linalg::faer_ndarray::{fast_ab, fast_atb};
    let (n, k) = y.dim();
    let gram = fast_atb(y, y);
    let (evals, v) = jacobi_symmetric(&gram)?;
    let yv = fast_ab(y, &v);
    let mut sigma = Array1::<f64>::zeros(k);
    let mut u = Array2::<f64>::zeros((n, k));
    for j in 0..k {
        sigma[j] = evals[j].max(0.0).sqrt();
        if sigma[j] > GEOMETRY_EPS {
            let inv_sigma = 1.0 / sigma[j];
            for i in 0..n {
                u[[i, j]] = yv[[i, j]] * inv_sigma;
            }
        }
    }
    Ok((u, sigma, v))
}

/// Frobenius-nearest matrix with orthonormal columns: the orthogonal factor `U`
/// of the polar decomposition `Y = U P`, with `UᵀU = Iₖ` and `P` symmetric
/// positive-definite. For a tall `Y` (`n × k`, `n ≥ k`) this is
/// `U = Y (YᵀY)^{-1/2}`, the orthogonal projection of `Y` onto the Stiefel
/// manifold `St(k, n)` — and hence onto the orthonormal-frame representative
/// set of the Grassmannian `Gr(k, n)` — under the Frobenius norm. So
/// `‖Y − U‖_F` is the *exact* distance from `Y` to that manifold. This is the
/// genuine nearest point, unlike the QR retraction `qf(Y)`, which only agrees
/// to first order.
///
/// Computed as `U Vᵀ` from the thin SVD `Y = U Σ Vᵀ` ([`thin_svd_gram`]), which
/// equals `Y (YᵀY)^{-1/2}` and shares the Gram eigendecomposition with the
/// geodesic SVD. A non-positive Gram eigenvalue (`σ² ≤ GEOMETRY_EPS`) means `Y`
/// has rank `< k`, where the nearest orthonormal frame is not unique; that is
/// surfaced as a [`GeometryError::Singular`] rather than returning an arbitrary
/// frame.
pub(crate) fn polar_factor(y: &Array2<f64>) -> GeometryResult<Array2<f64>> {
    let (n, k) = y.dim();
    if n < k {
        return Err(GeometryError::InvalidPoint(
            "polar factor requires a tall matrix (n >= k)",
        ));
    }
    if !y.iter().all(|v| v.is_finite()) {
        return Err(GeometryError::InvalidPoint(
            "polar factor requires finite entries",
        ));
    }
    let (u, sigma, v) = thin_svd_gram(y)?;
    if !sigma.iter().all(|&s| s * s > GEOMETRY_EPS) {
        return Err(GeometryError::Singular(
            "polar factor is undefined for a rank-deficient matrix",
        ));
    }
    Ok(u.dot(&v.t()))
}

/// Dense matrix exponential `exp(A)` via scaling-and-squaring with a truncated
/// Taylor series. The Frobenius norm of `A` is driven below 1/4 by repeated
/// halving (`A → A / 2^s`), where Taylor converges rapidly and stably; the
/// result is then squared `s` times. With the scaled norm `θ < 1/4`, the
/// degree-12 Taylor tail is bounded by `θ^{13} / 13! · 1/(1 - θ)`; since `13! ≈
/// 6.23e9`, this is below `4·0.25^{13}/6.23e9 ≈ 3.8e-18`, i.e. under one f64 ulp,
/// so the fixed degree truly reaches full f64 precision (the `< 1/2` threshold
/// previously used left a ~2e-14 tail, two orders above an ulp). This is the
/// standard, exact algorithm; no eigendecomposition is assumed (the inputs here
/// are the non-normal canonical-metric block matrices on Stiefel, which are
/// skew-like but not symmetric, so `spectral_map_*` does not apply).
pub(crate) fn matrix_exp(a: &Array2<f64>) -> GeometryResult<Array2<f64>> {
    let n = a.nrows();
    if n != a.ncols() {
        return Err(GeometryError::InvalidPoint(
            "matrix exponential requires square input",
        ));
    }
    if !a.iter().all(|v| v.is_finite()) {
        return Err(GeometryError::InvalidPoint(
            "matrix exponential requires finite entries",
        ));
    }
    // Frobenius norm; choose the squaring count so the scaled matrix has norm
    // below 1/4, which keeps the degree-12 Taylor truncation under one f64 ulp.
    let mut frob = 0.0;
    for v in a.iter() {
        frob += v * v;
    }
    let frob = frob.sqrt();
    let squarings = if frob > 0.25 {
        (frob / 0.25).log2().ceil() as i32
    } else {
        0
    };
    let scale = 2.0_f64.powi(squarings);
    let a_scaled = a / scale;

    // exp(A_scaled) = sum_{k>=0} A_scaled^k / k! by term recurrence:
    //   term_k = term_{k-1} · A_scaled / k.
    // Both the Taylor term recurrence and the scaling-and-squaring use dense
    // n×n products; GPU-dispatch them via fast_ab for large blocks.
    use crate::linalg::faer_ndarray::fast_ab;
    let mut result = identity(n);
    let mut term = identity(n);
    for k in 1..=12 {
        term = fast_ab(&term, &a_scaled) / (k as f64);
        result = result + &term;
    }
    // exp(A) = exp(A_scaled)^{2^squarings}.
    for _ in 0..squarings {
        result = fast_ab(&result, &result);
    }
    Ok(result)
}

/// Cholesky factor `L` of a symmetric positive-definite matrix (`A = L Lᵀ`).
///
/// This is a *positive-definiteness* test, not a conditioning test: a genuine
/// SPD matrix with tiny eigenvalues (e.g. `[[1e-16]]`) must factor
/// successfully. A pivot is rejected only when it is non-finite or fails to be
/// strictly positive *relative to the matrix scale*. The floor
/// `GEOMETRY_EPS · max(1, trace(A)/n)` is the ambient scale of the matrix
/// multiplied by the relative machine-noise tolerance, so a positive pivot that
/// is merely small in absolute terms (but large relative to nothing — the whole
/// matrix is small) passes, while a zero, negative, or numerically-noise pivot
/// (indefinite / singular directions) is rejected.
///
/// Callers needing a *conditioning* margin (a lower bound on the smallest
/// eigenvalue) must check that separately; overloading this PD test with an
/// absolute `GEOMETRY_EPS` floor wrongly rejected well-formed small-scale SPD
/// points. No current caller (only `SpdManifold::matrix`, which validates SPD
/// membership) depends on a conditioning margin here.
pub(crate) fn cholesky_spd(a: &Array2<f64>) -> GeometryResult<Array2<f64>> {
    let n = a.nrows();
    if n != a.ncols() {
        return Err(GeometryError::InvalidPoint(
            "Cholesky requires square input",
        ));
    }
    // Scale-relative positive-definiteness floor. `trace(A)/n` is the mean
    // diagonal, which equals `mean(eigenvalues)` and is therefore the natural
    // scale of an SPD matrix's spectrum. The acceptance floor scales WITH the
    // matrix (it shrinks for tiny matrices), so a uniformly small but genuine
    // SPD matrix like `[[1e-16]]` — scale 1e-16, floor GEOMETRY_EPS·1e-16 =
    // 1e-28 — passes, while a pivot that has collapsed to numerical noise
    // relative to the matrix's own scale (the indefinite/singular directions)
    // is rejected. An absolute `GEOMETRY_EPS` floor would have wrongly rejected
    // such tiny SPD matrices; clamping the floor up to a constant would do the
    // same, so we deliberately let it shrink with the spectrum.
    let mut trace = 0.0_f64;
    for i in 0..n {
        trace += a[[i, i]];
    }
    if !trace.is_finite() {
        return Err(GeometryError::InvalidPoint(
            "matrix is not positive definite",
        ));
    }
    // Reference scale of the matrix's spectrum. The acceptance floor is this
    // scale times the relative tolerance, so a uniformly-tiny SPD matrix (small
    // scale) has a correspondingly tiny floor and still factors, while a pivot
    // that has collapsed to noise *relative to the matrix's own scale* (the
    // indefinite/singular case) is rejected.
    let scale = (trace / n as f64).abs().max(f64::MIN_POSITIVE);
    let scale_eps = GEOMETRY_EPS * scale;
    let mut l = Array2::<f64>::zeros((n, n));
    for i in 0..n {
        for j in 0..=i {
            let mut sum = a[[i, j]];
            for k in 0..j {
                sum -= l[[i, k]] * l[[j, k]];
            }
            if i == j {
                if !sum.is_finite() || sum <= scale_eps {
                    return Err(GeometryError::InvalidPoint(
                        "matrix is not positive definite",
                    ));
                }
                l[[i, j]] = sum.sqrt();
            } else {
                l[[i, j]] = sum / l[[j, j]];
            }
        }
    }
    Ok(l)
}

#[cfg(test)]
mod cholesky_tests {
    use super::{GeometryError, cholesky_spd};
    use ndarray::Array2;

    /// A genuine SPD matrix with a uniformly tiny spectrum (`[[1e-16]]`) must
    /// factor: the issue is positive-definiteness, not absolute scale. The old
    /// absolute `GEOMETRY_EPS` floor wrongly rejected it.
    #[test]
    fn cholesky_accepts_tiny_spd() {
        let mut a = Array2::<f64>::zeros((1, 1));
        a[[0, 0]] = 1.0e-16;
        let l = cholesky_spd(&a).expect("tiny positive 1x1 must be SPD");
        assert!((l[[0, 0]] - 1.0e-8).abs() <= 1.0e-16);
    }

    /// A well-scaled SPD matrix factors and reproduces `L Lᵀ = A`.
    #[test]
    fn cholesky_accepts_well_scaled_spd() {
        // [[4, 2], [2, 3]] is SPD (eigenvalues ≈ 5.56, 1.44).
        let mut a = Array2::<f64>::zeros((2, 2));
        a[[0, 0]] = 4.0;
        a[[0, 1]] = 2.0;
        a[[1, 0]] = 2.0;
        a[[1, 1]] = 3.0;
        let l = cholesky_spd(&a).expect("well-scaled SPD must factor");
        let recon = l.dot(&l.t());
        for i in 0..2 {
            for j in 0..2 {
                assert!(
                    (recon[[i, j]] - a[[i, j]]).abs() <= 1.0e-12,
                    "L Lᵀ != A at ({i},{j})"
                );
            }
        }
    }

    /// A zero pivot (singular) and an indefinite matrix must be rejected as not
    /// positive definite — the scale-relative floor still catches the genuine
    /// non-PD case.
    #[test]
    fn cholesky_rejects_zero_and_indefinite() {
        let zero = Array2::<f64>::zeros((1, 1));
        match cholesky_spd(&zero) {
            Err(GeometryError::InvalidPoint(_)) => {}
            other => panic!("expected non-PD rejection of zero pivot, got {other:?}"),
        }
        // [[1, 2], [2, 1]] has eigenvalues 3 and −1 (indefinite): the Schur
        // complement pivot 1 − 4 = −3 is negative.
        let mut indef = Array2::<f64>::zeros((2, 2));
        indef[[0, 0]] = 1.0;
        indef[[0, 1]] = 2.0;
        indef[[1, 0]] = 2.0;
        indef[[1, 1]] = 1.0;
        match cholesky_spd(&indef) {
            Err(GeometryError::InvalidPoint(_)) => {}
            other => panic!("expected non-PD rejection of indefinite matrix, got {other:?}"),
        }
    }
}

#[cfg(test)]
mod qr_thin_tests {
    use super::qr_thin;
    use ndarray::Array2;

    /// Two identical columns make the second residual vanish; the fallback axis
    /// must be Gram–Schmidted against the first accepted column so `QᵀQ = I`.
    /// The old behavior planted `e₂` directly, giving `q₁·q₂ = 1/√2`.
    #[test]
    fn qr_thin_duplicated_columns_orthonormal() {
        let mut a = Array2::<f64>::zeros((2, 2));
        // Both columns = (1, 1).
        a[[0, 0]] = 1.0;
        a[[1, 0]] = 1.0;
        a[[0, 1]] = 1.0;
        a[[1, 1]] = 1.0;
        let (q, r) = qr_thin(&a);
        // Deficient second column ⇒ R[1,1] = 0.
        assert!(
            r[[1, 1]].abs() <= 1.0e-14,
            "deficient column must set R[1,1]=0"
        );
        let gram = q.t().dot(&q);
        for i in 0..2 {
            for j in 0..2 {
                let want = if i == j { 1.0 } else { 0.0 };
                assert!(
                    (gram[[i, j]] - want).abs() <= 1.0e-12,
                    "QᵀQ != I at ({i},{j}): got {}",
                    gram[[i, j]]
                );
            }
        }
    }

    /// A full-rank input still gives `QᵀQ = I` and reconstructs `A = QR`.
    #[test]
    fn qr_thin_full_rank_reconstructs() {
        let mut a = Array2::<f64>::zeros((3, 2));
        a[[0, 0]] = 1.0;
        a[[1, 0]] = 1.0;
        a[[2, 0]] = 0.0;
        a[[0, 1]] = 1.0;
        a[[1, 1]] = 0.0;
        a[[2, 1]] = 1.0;
        let (q, r) = qr_thin(&a);
        let gram = q.t().dot(&q);
        for i in 0..2 {
            for j in 0..2 {
                let want = if i == j { 1.0 } else { 0.0 };
                assert!(
                    (gram[[i, j]] - want).abs() <= 1.0e-12,
                    "QᵀQ != I at ({i},{j})"
                );
            }
        }
        let recon = q.dot(&r);
        for i in 0..3 {
            for j in 0..2 {
                assert!(
                    (recon[[i, j]] - a[[i, j]]).abs() <= 1.0e-12,
                    "QR != A at ({i},{j})"
                );
            }
        }
    }
}

#[cfg(test)]
mod jacobi_tests {
    use super::{GeometryError, jacobi_symmetric};
    use ndarray::Array2;

    /// A large-norm SPD matrix has off-diagonal residuals after
    /// diagonalization that scale with `||A||_F`, so they sit far above the
    /// old *absolute* `1e-13` cutoff even when the decomposition is, in fact,
    /// fully converged. The relative threshold (`1e-13 * ||A||_F`) recognizes
    /// convergence here and returns the correct spectrum instead of grinding
    /// through `max_iter` sweeps and silently returning a partial diagonal.
    #[test]
    fn jacobi_converges_on_large_norm_spd() {
        // Q diag(1e8, 2e8, 3e8) Qᵀ for an orthogonal Q built from a planar
        // rotation in the (0,1) plane; eigenvalues are huge so the matrix
        // norm is ~1e8 and any absolute 1e-13 off-diagonal test is hopeless.
        let theta = 0.7_f64;
        let (c, s) = (theta.cos(), theta.sin());
        let mut q = Array2::<f64>::eye(3);
        q[[0, 0]] = c;
        q[[0, 1]] = -s;
        q[[1, 0]] = s;
        q[[1, 1]] = c;
        let lambda = [1.0e8_f64, 2.0e8, 3.0e8];
        let mut diag = Array2::<f64>::zeros((3, 3));
        for i in 0..3 {
            diag[[i, i]] = lambda[i];
        }
        let a = q.dot(&diag).dot(&q.t());

        let (evals, evecs) = jacobi_symmetric(&a).expect("large-norm SPD must converge");
        let mut sorted: Vec<f64> = evals.to_vec();
        sorted.sort_by(|x, y| x.partial_cmp(y).unwrap());
        for (got, want) in sorted.iter().zip(lambda.iter()) {
            assert!(
                (got - want).abs() <= 1.0e-6 * want,
                "eigenvalue mismatch: got {got}, want {want}"
            );
        }
        // V diag(evals) Vᵀ must reconstruct A (relative to its scale).
        let mut diag_e = Array2::<f64>::zeros((3, 3));
        for i in 0..3 {
            diag_e[[i, i]] = evals[i];
        }
        let recon = evecs.dot(&diag_e).dot(&evecs.t());
        for i in 0..3 {
            for j in 0..3 {
                assert!(
                    (recon[[i, j]] - a[[i, j]]).abs() <= 1.0e-6 * 3.0e8,
                    "reconstruction mismatch at ({i},{j})"
                );
            }
        }
    }

    /// A clustered/degenerate spectrum (two coincident eigenvalues) must still
    /// converge and reproduce the multiplicity. This guards against the
    /// relative threshold being so tight that ordinary near-degenerate SPD
    /// inputs trip the new non-convergence error.
    #[test]
    fn jacobi_handles_clustered_spectrum() {
        // diag(5, 5, 1) rotated in the (0,2) plane; the degenerate pair stays
        // degenerate under rotation.
        let theta = 0.4_f64;
        let (c, s) = (theta.cos(), theta.sin());
        let mut q = Array2::<f64>::eye(3);
        q[[0, 0]] = c;
        q[[0, 2]] = -s;
        q[[2, 0]] = s;
        q[[2, 2]] = c;
        let lambda = [5.0_f64, 5.0, 1.0];
        let mut diag = Array2::<f64>::zeros((3, 3));
        for i in 0..3 {
            diag[[i, i]] = lambda[i];
        }
        let a = q.dot(&diag).dot(&q.t());

        let (evals, evecs) = jacobi_symmetric(&a).expect("clustered SPD must converge");
        let mut sorted: Vec<f64> = evals.to_vec();
        sorted.sort_by(|x, y| x.partial_cmp(y).unwrap());
        assert!((sorted[0] - 1.0).abs() <= 1.0e-12);
        assert!((sorted[1] - 5.0).abs() <= 1.0e-12);
        assert!((sorted[2] - 5.0).abs() <= 1.0e-12);
        // Eigenvectors must remain orthonormal even across the degenerate pair.
        let gram = evecs.t().dot(&evecs);
        for i in 0..3 {
            for j in 0..3 {
                let want = if i == j { 1.0 } else { 0.0 };
                assert!(
                    (gram[[i, j]] - want).abs() <= 1.0e-12,
                    "eigenvectors not orthonormal at ({i},{j})"
                );
            }
        }
    }

    /// Non-convergence must now surface as `GeometryError::Singular` instead
    /// of a silently-returned partial diagonal. A symmetric input carrying a
    /// non-finite off-diagonal can never drive the largest off-diagonal
    /// magnitude below `1e-13 * ||A||_F` (the norm itself is non-finite), so
    /// the sweep exhausts `max_iter` and the solver must error rather than
    /// hand back the un-diagonalized matrix's diagonal.
    #[test]
    fn jacobi_errors_on_non_convergence() {
        let mut a = Array2::<f64>::eye(3);
        a[[0, 1]] = f64::NAN;
        a[[1, 0]] = f64::NAN;
        match jacobi_symmetric(&a) {
            Err(GeometryError::Singular(_)) => {}
            other => panic!("expected Singular non-convergence error, got {other:?}"),
        }
    }
}
