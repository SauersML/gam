//! User-selectable response geometries beyond Sphere and Simplex.
//!
//! The fit DSL exposes `response_geometry="..."`: one scalar Gaussian GAM is
//! fitted per tangent coordinate at a fixed base point (the intrinsic Fréchet
//! mean when none is supplied), and predictions are mapped back to the manifold
//! by the exponential map. Sphere and Simplex have bespoke batched wrappers in
//! their own modules; this module supplies the same `(values 2-D, base 1-D) →
//! tangent 2-D` / `(tangent 2-D, base 1-D) → values 2-D` contract for the
//! curved matrix manifolds whose per-point math is already wired in
//! [`crate::geometry`] but which were never reachable as a *fittable* response
//! geometry: the SPD cone `Sym⁺(n)`, the Grassmannian `Gr(k, n)`, the Stiefel
//! manifold `St(k, n)`, and the Poincaré ball `B^d_κ`.
//!
//! Every primitive here delegates to the canonical landed math
//! ([`RiemannianManifold::exp_map`]/[`log_map`](RiemannianManifold::log_map) and
//! the Poincaré [`exp_map`](crate::manifolds::poincare::exp_map)/[`log_map`](crate::manifolds::poincare::log_map));
//! the only new code is the batched row loop, the base-point dimension wiring,
//! and a generic Riemannian Karcher (Fréchet) mean shared by all four. There is
//! no separate per-manifold mean: the SPD safeguarded Karcher iteration is
//! generalised once, over the metric supplied by
//! [`RiemannianManifold::metric_tensor`], so adding a curved response geometry
//! is a single resolver arm.

use ndarray::{Array1, Array2, ArrayView1, ArrayView2};

use crate::manifold::{
    GEOMETRY_EPS, RiemannianManifold, flatten, from_flat, jacobi_symmetric, spectral_map_symmetric,
    sym,
};
use crate::manifolds::constant_curvature::ConstantCurvature;
use crate::{GeometryError, GeometryResult, GrassmannManifold, SpdManifold, StiefelManifold};

/// Split a parenthesised `key=value, key=value` parameter list into trimmed,
/// lower-cased `(key, value)` pairs. An empty list is valid (`spd()`).
fn parse_kv(inner: &str) -> Result<Vec<(String, String)>, String> {
    let trimmed = inner.trim();
    if trimmed.is_empty() {
        return Ok(Vec::new());
    }
    let mut out = Vec::new();
    for piece in trimmed.split(',') {
        let piece = piece.trim();
        if piece.is_empty() {
            continue;
        }
        let (k, v) = piece
            .split_once('=')
            .ok_or_else(|| format!("response_geometry parameter {piece:?} must be key=value"))?;
        out.push((k.trim().to_ascii_lowercase(), v.trim().to_string()));
    }
    Ok(out)
}

/// A fittable curved response geometry. Each variant carries the shape the user
/// requested; the embedding/ambient flat dimension is fixed by that shape and
/// is the column count of the `values` matrix the caller supplies.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum ResponseManifold {
    /// Symmetric positive-definite `n×n` matrices, flattened row-major to `n²`
    /// ambient coordinates (the layout [`SpdManifold`] uses).
    Spd { n: usize },
    /// `k`-dimensional subspaces of `ℝⁿ`, represented by an orthonormal `n×k`
    /// frame flattened to `n·k` ambient coordinates.
    Grassmann { k: usize, n: usize },
    /// Orthonormal `k`-frames in `ℝⁿ`, flattened to `n·k` ambient coordinates.
    Stiefel { k: usize, n: usize },
    /// The Poincaré ball of dimension `d` with curvature `κ < 0`.
    Poincare { dim: usize, curvature: f64 },
    /// Constant-curvature manifold `M_κ` of dimension `d` with curvature `κ`
    /// (any finite real value). `κ > 0` → spherical, `κ = 0` → flat (Euclidean
    /// up to scale), `κ < 0` → hyperbolic (Poincaré ball). Unlike `Poincare`,
    /// which fixes `κ < 0`, this variant accepts any curvature including zero
    /// and positive values, and is the target for curvature-as-estimand fits
    /// where `κ̂` is optimized over all of ℝ (#1104).
    ConstantCurvature { dim: usize, kappa: f64 },
}

impl ResponseManifold {
    /// Resolve a lower-cased geometry label and its shape parameters into a
    /// response manifold. Shape parameters are passed positionally exactly as
    /// the FFI marshals them; absent/zero values are rejected here so the error
    /// surfaces at selection time rather than mid-fit.
    ///
    /// - `"spd"` needs `n` (matrix side).
    /// - `"grassmann"` / `"stiefel"` need `k` and `n` with `1 ≤ k ≤ n`.
    /// - `"poincare"` needs `dim` and a strictly negative `curvature`.
    pub fn resolve(
        kind: &str,
        n: Option<usize>,
        k: Option<usize>,
        dim: Option<usize>,
        curvature: Option<f64>,
    ) -> Result<Self, String> {
        match kind {
            "spd" => {
                let n = n.ok_or_else(|| "response_geometry='spd' requires n".to_string())?;
                if n == 0 {
                    return Err("response_geometry='spd' requires n >= 1".to_string());
                }
                Ok(Self::Spd { n })
            }
            "grassmann" => {
                let k = k.ok_or_else(|| "response_geometry='grassmann' requires k".to_string())?;
                let n = n.ok_or_else(|| "response_geometry='grassmann' requires n".to_string())?;
                if k == 0 || n == 0 || k > n {
                    return Err("response_geometry='grassmann' requires 1 <= k <= n".to_string());
                }
                Ok(Self::Grassmann { k, n })
            }
            "stiefel" => {
                let k = k.ok_or_else(|| "response_geometry='stiefel' requires k".to_string())?;
                let n = n.ok_or_else(|| "response_geometry='stiefel' requires n".to_string())?;
                if k == 0 || n == 0 || k > n {
                    return Err("response_geometry='stiefel' requires 1 <= k <= n".to_string());
                }
                Ok(Self::Stiefel { k, n })
            }
            "poincare" => {
                let dim =
                    dim.ok_or_else(|| "response_geometry='poincare' requires dim".to_string())?;
                if dim == 0 {
                    return Err("response_geometry='poincare' requires dim >= 1".to_string());
                }
                let curvature = curvature
                    .ok_or_else(|| "response_geometry='poincare' requires curvature".to_string())?;
                if !(curvature.is_finite() && curvature < 0.0) {
                    return Err(
                        "response_geometry='poincare' requires finite curvature < 0".to_string()
                    );
                }
                Ok(Self::Poincare { dim, curvature })
            }
            "constant_curvature" => {
                let dim = dim.ok_or_else(|| {
                    "response_geometry='constant_curvature' requires dim".to_string()
                })?;
                if dim == 0 {
                    return Err(
                        "response_geometry='constant_curvature' requires dim >= 1".to_string()
                    );
                }
                // curvature defaults to 0 (flat) when not supplied — the user can
                // supply any finite value; the κ-estimand outer loop will optimize it.
                let kappa = curvature.unwrap_or(0.0);
                if !kappa.is_finite() {
                    return Err(
                        "response_geometry='constant_curvature' requires finite curvature"
                            .to_string(),
                    );
                }
                Ok(Self::ConstantCurvature { dim, kappa })
            }
            other => Err(format!(
                "response_geometry must be one of 'spd', 'grassmann', 'stiefel', 'poincare', \
                 'constant_curvature', 'spherical', or 'simplex'; got {other:?}"
            )),
        }
    }

    /// Parse a user-facing `response_geometry` label, magic-by-default: the head
    /// is the geometry name, an optional parenthesised `key=value` list carries
    /// shape parameters, and anything not given is inferred from the ambient
    /// column count `cols` of the response matrix.
    ///
    /// Recognised forms (case-insensitive, whitespace tolerant):
    /// - `"spd"` — `n = √cols` (must be a perfect square).
    /// - `"grassmann(k=2)"` or `"grassmann(k=2,n=5)"` — `n` defaults to
    ///   `cols / k`; `k` is required (it cannot be inferred from `n·k`).
    /// - `"stiefel(k=2)"` / `"stiefel(k=2,n=5)"` — same inference as Grassmann.
    /// - `"poincare"` or `"poincare(curvature=-0.5)"` — `dim = cols`; curvature
    ///   defaults to `-1.0`.
    ///
    /// This is the single mapping from the formula-DSL string to a constructed
    /// response manifold; the FFI passes the raw label straight through.
    pub fn parse(label: &str, cols: usize) -> Result<Self, String> {
        let lowered = label.trim().to_ascii_lowercase();
        let (head, params) = match lowered.split_once('(') {
            Some((h, rest)) => {
                let rest = rest.trim_end();
                let inner = rest
                    .strip_suffix(')')
                    .ok_or_else(|| format!("response_geometry {label:?}: missing closing ')'"))?;
                (h.trim().to_string(), parse_kv(inner)?)
            }
            None => (lowered.clone(), Vec::new()),
        };
        let get_usize = |key: &str| -> Result<Option<usize>, String> {
            for (k, v) in &params {
                if k == key {
                    let parsed: usize = v.parse().map_err(|_| {
                        format!("response_geometry {label:?}: {key} must be a non-negative integer")
                    })?;
                    return Ok(Some(parsed));
                }
            }
            Ok(None)
        };
        let get_f64 = |key: &str| -> Result<Option<f64>, String> {
            for (k, v) in &params {
                if k == key {
                    let parsed: f64 = v.parse().map_err(|_| {
                        format!("response_geometry {label:?}: {key} must be a real number")
                    })?;
                    return Ok(Some(parsed));
                }
            }
            Ok(None)
        };

        match head.as_str() {
            "spd" => {
                let n = match get_usize("n")? {
                    Some(n) => n,
                    None => {
                        let r = (cols as f64).sqrt().round() as usize;
                        if r * r != cols {
                            return Err(format!(
                                "response_geometry='spd': {cols} response columns is not a perfect \
                                 square; pass spd(n=...) explicitly"
                            ));
                        }
                        r
                    }
                };
                Self::resolve("spd", Some(n), None, None, None)
            }
            "grassmann" | "stiefel" => {
                let k = get_usize("k")?.ok_or_else(|| {
                    format!("response_geometry='{head}' requires k, e.g. {head}(k=2)")
                })?;
                let n = match get_usize("n")? {
                    Some(n) => n,
                    None => {
                        if k == 0 || cols % k != 0 {
                            return Err(format!(
                                "response_geometry='{head}': {cols} response columns is not \
                                 divisible by k={k}; pass {head}(k=..,n=..) explicitly"
                            ));
                        }
                        cols / k
                    }
                };
                Self::resolve(&head, Some(n), Some(k), None, None)
            }
            "poincare" => {
                let dim = get_usize("dim")?.unwrap_or(cols);
                let curvature = get_f64("curvature")?.unwrap_or(-1.0);
                Self::resolve("poincare", None, None, Some(dim), Some(curvature))
            }
            "constant_curvature" => {
                let dim = get_usize("dim")?.unwrap_or(cols);
                // κ defaults to 0 (flat initial point for the REML optimizer).
                let kappa = get_f64("kappa")?
                    .or_else(|| get_f64("curvature").ok().flatten())
                    .unwrap_or(0.0);
                Self::resolve("constant_curvature", None, None, Some(dim), Some(kappa))
            }
            other => Err(format!(
                "response_geometry must be one of 'spd', 'grassmann(k=..)', 'stiefel(k=..)', \
                 'poincare', 'constant_curvature', 'spherical', or 'simplex'; got {other:?}"
            )),
        }
    }

    /// Canonical, fully-specified label echoed back to the caller (mirrors the
    /// way the sphere/simplex dispatch reports its resolved coordinate label).
    pub fn canonical_label(&self) -> String {
        match self {
            Self::Spd { n } => format!("spd(n={n})"),
            Self::Grassmann { k, n } => format!("grassmann(k={k},n={n})"),
            Self::Stiefel { k, n } => format!("stiefel(k={k},n={n})"),
            Self::Poincare { dim, curvature } => {
                format!("poincare(dim={dim},curvature={curvature})")
            }
            Self::ConstantCurvature { dim, kappa } => {
                format!("constant_curvature(dim={dim},kappa={kappa})")
            }
        }
    }

    /// Ambient (flattened) coordinate count: the column width of the `values`
    /// matrix and the `base` vector.
    pub fn ambient_dim(&self) -> usize {
        match self {
            Self::Spd { n } => n * n,
            Self::Grassmann { k, n } | Self::Stiefel { k, n } => n * k,
            Self::Poincare { dim, .. } | Self::ConstantCurvature { dim, .. } => *dim,
        }
    }

    /// Whether the intrinsic Fréchet (Karcher) mean of manifold-valued responses
    /// can have MORE THAN ONE local minimizer — i.e. whether the geometry is
    /// positively curved somewhere, so a single safeguarded descent can stall in
    /// a suboptimal basin and a multistart is worth the extra restarts.
    ///
    /// - `Spd` and `Poincare` are Hadamard manifolds (globally non-positively
    ///   curved / CAT(0)): the dispersion `V(P)=Σ wᵢ d(P,Xᵢ)²` is globally,
    ///   strictly geodesically convex, so the Fréchet mean is unique and one
    ///   descent reaches it — no restart can improve on it.
    /// - `Stiefel` (`St(n,1)=Sⁿ⁻¹`, and `k>1` with the canonical metric) and
    ///   `Grassmann` (`Gr(1,n)=ℝPⁿ⁻¹`) carry non-negative sectional curvature:
    ///   for a cloud whose spread approaches the injectivity radius the mean is
    ///   only *locally* unique and several basins can coexist, so a
    ///   different-seed restart can land a strictly lower-dispersion mean.
    /// - `ConstantCurvature` is positively curved exactly when `κ > 0`
    ///   (spherical); `κ ≤ 0` is flat/hyperbolic and Hadamard.
    fn frechet_mean_can_be_nonunique(&self) -> bool {
        match self {
            Self::Spd { .. } | Self::Poincare { .. } => false,
            Self::Grassmann { .. } | Self::Stiefel { .. } => true,
            Self::ConstantCurvature { kappa, .. } => *kappa > 0.0,
        }
    }

    /// Build the underlying [`RiemannianManifold`] for the matrix geometries.
    /// `None` for Poincaré, whose primitives are free functions parameterised
    /// by curvature rather than a trait object.
    fn riemannian(&self) -> Option<Box<dyn RiemannianManifold>> {
        match self {
            Self::Spd { n } => Some(Box::new(SpdManifold::new(*n))),
            Self::Grassmann { k, n } => GrassmannManifold::new(*k, *n)
                .ok()
                .map(|m| Box::new(m) as _),
            Self::Stiefel { k, n } => StiefelManifold::new(*k, *n).ok().map(|m| Box::new(m) as _),
            Self::ConstantCurvature { dim, kappa } => {
                Some(Box::new(ConstantCurvature::new(*dim, *kappa)))
            }
            Self::Poincare { .. } => None,
        }
    }

    /// Per-point logarithm `log_base(value)` in flat ambient coordinates.
    fn log_point(
        &self,
        base: ArrayView1<'_, f64>,
        value: ArrayView1<'_, f64>,
    ) -> GeometryResult<Array1<f64>> {
        match self {
            Self::Poincare { curvature, .. } => {
                crate::manifolds::poincare::log_map(base, value, *curvature)
            }
            // ConstantCurvature implements RiemannianManifold::log_map directly.
            Self::ConstantCurvature { .. }
            | Self::Spd { .. }
            | Self::Grassmann { .. }
            | Self::Stiefel { .. } => self
                .riemannian()
                .expect("riemannian response manifold")
                .log_map(base, value),
        }
    }

    /// Per-point exponential `exp_base(tangent)` in flat ambient coordinates.
    fn exp_point(
        &self,
        base: ArrayView1<'_, f64>,
        tangent: ArrayView1<'_, f64>,
    ) -> GeometryResult<Array1<f64>> {
        match self {
            Self::Poincare { curvature, .. } => {
                crate::manifolds::poincare::exp_map(base, tangent, *curvature)
            }
            Self::ConstantCurvature { .. }
            | Self::Spd { .. }
            | Self::Grassmann { .. }
            | Self::Stiefel { .. } => self
                .riemannian()
                .expect("riemannian response manifold")
                .exp_map(base, tangent),
        }
    }

    /// Euclidean / Frobenius distance from an arbitrary ambient row to the
    /// candidate response geometry, in flat ambient coordinates — the extrinsic
    /// constraint-violation distance behind [`response_projection_residual`].
    ///
    /// Unlike [`log_point`](Self::log_point), which is gatekept to *genuine*
    /// manifold points on both arguments, this accepts off-manifold `value`. The
    /// distance is computed in closed form per geometry and is **well-defined for
    /// every input** — there is no rank-deficiency error path, because the
    /// distance to a set is defined even where the nearest point is not unique:
    ///
    /// * `Gr(k, n)` / `St(k, n)` — distance to the orthonormal-frame set,
    ///   `√Σ_i (σ_i − 1)²` with `σ_i = √max(λ_i(YᵀY), 0)` the singular values of
    ///   the `n × k` frame `Y`. Exact for every rank (`σ_i = 0` columns
    ///   contribute `1` each). Grassmann and Stiefel coincide because this module
    ///   represents Grassmann points by frames — it is a *representation*
    ///   distance, not a subspace/principal-angle distance.
    /// * SPD cone — distance to the *closed* PSD cone,
    ///   `√(‖skew(A)‖_F² + Σ_{λ_i<0} λ_i²)` with `λ_i` the eigenvalues of the
    ///   symmetric part `sym(A)`. This is the infimum distance to the open SPD
    ///   cone; a zero distance means PSD, **not** strictly PD.
    /// * Poincaré ball — distance to the *manifold* open ball of radius
    ///   `R = 1/√(−c)`: `max(0, ‖x‖ − R)`. (This uses the true radius `R`, not
    ///   the slightly smaller numerical safety radius used when projecting points
    ///   for a fit, so interior points score exactly zero.)
    /// * `ConstantCurvature` — distance to the chart *domain*: `0` for `κ ≥ 0`
    ///   (chart is all of `ℝ^d`), else `max(0, ‖x‖ − 1/√(−κ))`. The curvature
    ///   lives in the metric, not the domain, so this is a domain-admissibility
    ///   check only and carries little curvature information.
    fn manifold_residual(&self, value: ArrayView1<'_, f64>) -> GeometryResult<f64> {
        match self {
            Self::Poincare { curvature, .. } => ball_domain_residual(value, *curvature),
            Self::ConstantCurvature { kappa, .. } => {
                if *kappa >= 0.0 {
                    Ok(0.0)
                } else {
                    ball_domain_residual(value, *kappa)
                }
            }
            Self::Spd { n } => {
                let mat = from_flat(value, *n, *n)?;
                let symm = sym(&mat);
                let psd = spectral_map_symmetric(&symm, |lam| Ok(lam.max(0.0)))?;
                // Distance to the closed PSD cone, measured against the original
                // (skew included) input so the skew-symmetric part is counted.
                Ok(frobenius_distance(value, flatten(&psd).view()))
            }
            Self::Grassmann { k, n } | Self::Stiefel { k, n } => {
                use gam_linalg::faer_ndarray::fast_atb;
                let frame = from_flat(value, *n, *k)?;
                let gram = fast_atb(&frame, &frame);
                let (evals, _) = jacobi_symmetric(&gram)?;
                let mut sq = 0.0_f64;
                for &lam in evals.iter() {
                    let sigma = lam.max(0.0).sqrt();
                    let d = sigma - 1.0;
                    sq += d * d;
                }
                Ok(sq.sqrt())
            }
        }
    }

    /// Squared metric norm `‖v‖²_base` of a tangent at `base`. Used by the
    /// Karcher iteration's stationarity test. Poincaré uses the conformal
    /// factor squared; the matrix manifolds and ConstantCurvature use the trait
    /// metric tensor.
    fn sq_metric_norm(
        &self,
        base: ArrayView1<'_, f64>,
        v: ArrayView1<'_, f64>,
    ) -> GeometryResult<f64> {
        match self {
            Self::Poincare { curvature, .. } => {
                let lam = crate::manifolds::poincare::conformal_factor(base, *curvature)?;
                Ok(lam * lam * v.iter().map(|x| x * x).sum::<f64>())
            }
            Self::ConstantCurvature { .. }
            | Self::Spd { .. }
            | Self::Grassmann { .. }
            | Self::Stiefel { .. } => {
                let g = self
                    .riemannian()
                    .expect("riemannian response manifold")
                    .metric_tensor(base)?;
                let gv = g.dot(&v);
                Ok(v.dot(&gv).max(0.0))
            }
        }
    }
}

/// Batched response-geometry logarithm: map every manifold-valued response row
/// to its tangent coordinate at `base`. `values` is `(n_rows, ambient)`, `base`
/// is `(ambient,)`, and the returned tangent is `(n_rows, ambient)` (the same
/// flat ambient layout — the tangent of a matrix manifold is itself a flattened
/// matrix). The scalar Gaussian GAMs the caller fits operate column-wise on
/// this matrix exactly as they do for the sphere.
pub fn response_log_map(
    manifold: ResponseManifold,
    values: ArrayView2<'_, f64>,
    base: ArrayView1<'_, f64>,
) -> Result<Array2<f64>, String> {
    let ambient = manifold.ambient_dim();
    let (n_rows, cols) = values.dim();
    if base.len() != ambient {
        return Err(format!(
            "response geometry base point has length {}; expected {ambient}",
            base.len()
        ));
    }
    if cols != ambient {
        return Err(format!(
            "response geometry values have {cols} columns; expected {ambient}"
        ));
    }
    let mut out = Array2::<f64>::zeros((n_rows, ambient));
    for row in 0..n_rows {
        let tangent = manifold
            .log_point(base, values.row(row))
            .map_err(|e| format!("response geometry log map (row {row}): {e}"))?;
        out.row_mut(row).assign(&tangent);
    }
    Ok(out)
}

/// Batched response-geometry exponential: map predicted tangent coordinates
/// back to manifold-valued responses at `base`. Inverse of [`response_log_map`]
/// with the same shapes.
pub fn response_exp_map(
    manifold: ResponseManifold,
    tangent: ArrayView2<'_, f64>,
    base: ArrayView1<'_, f64>,
) -> Result<Array2<f64>, String> {
    let ambient = manifold.ambient_dim();
    let (n_rows, cols) = tangent.dim();
    if base.len() != ambient {
        return Err(format!(
            "response geometry base point has length {}; expected {ambient}",
            base.len()
        ));
    }
    if cols != ambient {
        return Err(format!(
            "response geometry tangent has {cols} columns; expected {ambient}"
        ));
    }
    if !tangent.iter().all(|v| v.is_finite()) {
        return Err("response geometry tangent must contain only finite values".to_string());
    }
    let mut out = Array2::<f64>::zeros((n_rows, ambient));
    for row in 0..n_rows {
        let value = manifold
            .exp_point(base, tangent.row(row))
            .map_err(|e| format!("response geometry exp map (row {row}): {e}"))?;
        out.row_mut(row).assign(&value);
    }
    Ok(out)
}

/// Numerically-stable Euclidean norm `‖v‖₂`, scaled by the largest-magnitude
/// entry so the squared sum cannot overflow for large but finite inputs.
fn scaled_l2_norm(v: ArrayView1<'_, f64>) -> f64 {
    let mut scale = 0.0_f64;
    for &x in v.iter() {
        let a = x.abs();
        if a > scale {
            scale = a;
        }
    }
    if scale == 0.0 {
        return 0.0;
    }
    let mut ssq = 0.0_f64;
    for &x in v.iter() {
        let t = x / scale;
        ssq += t * t;
    }
    scale * ssq.sqrt()
}

/// Numerically-stable Frobenius distance `‖a − b‖₂` over equal-length flat
/// vectors, scaled by the largest entrywise difference to avoid overflow.
fn frobenius_distance(a: ArrayView1<'_, f64>, b: ArrayView1<'_, f64>) -> f64 {
    let mut scale = 0.0_f64;
    for (x, y) in a.iter().zip(b.iter()) {
        let d = (x - y).abs();
        if d > scale {
            scale = d;
        }
    }
    if scale == 0.0 {
        return 0.0;
    }
    let mut ssq = 0.0_f64;
    for (x, y) in a.iter().zip(b.iter()) {
        let t = (x - y) / scale;
        ssq += t * t;
    }
    scale * ssq.sqrt()
}

/// Distance from `value` to the open ball of radius `R = 1/√(−c)` (`c < 0`):
/// `max(0, ‖value‖ − R)`, the true Euclidean infimum distance to the ball.
/// Errors if the curvature is not a finite negative number.
fn ball_domain_residual(value: ArrayView1<'_, f64>, curvature: f64) -> GeometryResult<f64> {
    if !curvature.is_finite() || curvature >= 0.0 {
        return Err(GeometryError::InvalidPoint(
            "ball distance requires a finite negative curvature",
        ));
    }
    let radius = (-curvature).sqrt().recip();
    Ok((scaled_l2_norm(value) - radius).max(0.0))
}

/// Per-row extrinsic distance from ambient observations to a *candidate*
/// response geometry — a coordinate-dependent constraint / closure-distance
/// diagnostic.
///
/// What this is (and is not)
/// -------------------------
/// This is a cheap, pre-fit **constraint-violation** measure: given a candidate
/// response geometry, how far does each raw row sit from that geometry's
/// extrinsic representation (the unit-norm frame, the PSD cone, the Poincaré
/// ball)? It is **not** the post-fit on/off-manifold membership signal (which
/// comes from a fitted geometric smooth's residual and posterior predictive
/// density), and it is **not** a universal cross-geometry model-selection score:
/// it measures extrinsic constraint violation *in a chosen coordinate chart*,
/// not intrinsic topology or curvature. Different candidate geometries have
/// different chart codimensions (a full-dimensional Poincaré/`κ ≥ 0` chart can
/// score zero trivially), so residuals are not directly comparable across
/// candidates without a noise model and per-candidate calibration. Use it as a
/// fast per-candidate gate, with candidate-specific thresholds.
///
/// What it computes
/// ----------------
/// For each ambient row `x`, [`manifold_residual`](Self::manifold_residual)
/// returns the closed-form distance to the candidate geometry (well-defined for
/// every input and every rank — see that method for the per-geometry formulas),
/// and this returns:
///
/// * `residual[i]` — the absolute distance-to-geometry (zero for genuinely
///   admissible rows; for the matrix manifolds, exact to machine precision).
/// * `relative[i] = residual[i] / (‖x‖ + eps)` — the distance normalised by the
///   row's ambient magnitude. **Note:** this is dimensionless but *not*
///   scale-invariant for the fixed-radius geometries (Stiefel/Grassmann/ball)
///   and is *not* bounded by `1` (it diverges as `‖x‖ → 0`); it is scale-free
///   only for the homogeneous SPD cone. Treat it as `input_norm_relative`, not
///   an off-manifold fraction.
///
/// Unlike [`response_log_map`], **no base point is needed**. `values` is
/// `(n_rows, ambient)`; both returned arrays are `(n_rows,)`. Every fittable
/// response geometry — including `ConstantCurvature` — has a closed-form
/// distance, so no variant errors on a valid, finite input.
pub fn response_projection_residual(
    manifold: ResponseManifold,
    values: ArrayView2<'_, f64>,
) -> Result<(Array1<f64>, Array1<f64>), String> {
    let ambient = manifold.ambient_dim();
    let (n_rows, cols) = values.dim();
    if cols != ambient {
        return Err(format!(
            "response geometry values have {cols} columns; expected {ambient}"
        ));
    }
    if !values.iter().all(|v| v.is_finite()) {
        return Err("response geometry values must contain only finite values".to_string());
    }

    let mut residual = Array1::<f64>::zeros(n_rows);
    let mut relative = Array1::<f64>::zeros(n_rows);
    for row in 0..n_rows {
        let value = values.row(row);
        let dist = manifold
            .manifold_residual(value)
            .map_err(|e| format!("response geometry residual (row {row}): {e}"))?;
        let rel = dist / (scaled_l2_norm(value) + GEOMETRY_EPS);
        if !dist.is_finite() || !rel.is_finite() {
            return Err(format!(
                "response geometry residual (row {row}) is non-finite"
            ));
        }
        residual[row] = dist;
        relative[row] = rel;
    }
    Ok((residual, relative))
}

/// String-driven response-geometry log map: parse the user `label` (with shape
/// inference from the response column count), pick the base point (intrinsic
/// Fréchet mean when `base` is `None`), map every row to its tangent, and report
/// the canonical resolved label. This is the curved-manifold analogue of the
/// sphere/simplex dispatch and the single entry the FFI calls for these
/// geometries.
///
/// `weights` are the per-observation prior weights used ONLY to pick the intrinsic
/// base point (they are ignored when an explicit `base` is supplied). When the
/// caller supplies observation weights they must reach the linearization point so
/// the tangent chart is expanded around the *weighted* Fréchet mean — where the
/// weighted mass lives — matching the weighted tangent regression run there
/// (#2125). `None` recovers the uniform intrinsic mean.
pub fn dispatch_log_map(
    values: ArrayView2<'_, f64>,
    label: &str,
    base: Option<ArrayView1<'_, f64>>,
    weights: Option<ArrayView1<'_, f64>>,
) -> Result<(Array2<f64>, Array1<f64>, String), String> {
    let manifold = ResponseManifold::parse(label, values.ncols())?;
    let base_point = match base {
        Some(b) => b.to_owned(),
        None => response_frechet_mean(manifold, values, weights, 1.0e-12, 256)?,
    };
    let tangent = response_log_map(manifold, values, base_point.view())?;
    Ok((tangent, base_point, manifold.canonical_label()))
}

/// String-driven response-geometry exponential map: inverse of
/// [`dispatch_log_map`] given an explicit base point.
pub fn dispatch_exp_map(
    tangent: ArrayView2<'_, f64>,
    label: &str,
    base: ArrayView1<'_, f64>,
) -> Result<Array2<f64>, String> {
    let manifold = ResponseManifold::parse(label, tangent.ncols())?;
    response_exp_map(manifold, tangent, base)
}

/// Intrinsic (Karcher) Fréchet mean of manifold-valued responses, the default
/// base point when the user supplies none. `values` is `(n_rows, ambient)`.
///
/// This is the SPD safeguarded Karcher iteration generalised over an arbitrary
/// [`ResponseManifold`]: a Riemannian gradient-descent on the weighted
/// dispersion `V(P) = Σ_i w_i ‖log_P(X_i)‖²_P` with the descent direction
/// `ξ = Σ_i w_i log_P(X_i)` (`= −½ grad V`), a unit Karcher step `exp_P(t·ξ)`
/// with Armijo backtracking plus a round-off cushion, a best-iterate stall
/// guard, and the metric-norm stationarity test `‖ξ‖_P ≤ tol`. The SPD-specific
/// version in [`crate::manifolds::spd::spd_frechet_mean`] remains for the affine
/// inverse it caches per step; this generic form pays a metric-tensor solve but
/// covers all four geometries uniformly.
pub fn response_frechet_mean(
    manifold: ResponseManifold,
    values: ArrayView2<'_, f64>,
    weights: Option<ArrayView1<'_, f64>>,
    tol: f64,
    max_iter: usize,
) -> Result<Array1<f64>, String> {
    let ambient = manifold.ambient_dim();
    let (m, cols) = values.dim();
    if m == 0 || cols != ambient {
        return Err(format!(
            "response geometry Fréchet mean: values must be M×{ambient} with M >= 1"
        ));
    }
    if !(tol.is_finite() && tol > 0.0) {
        return Err("response geometry Fréchet mean tolerance must be finite and positive".into());
    }
    let w = crate::normalize_weights(m, weights)
        .map_err(|_| "response geometry Fréchet mean: invalid weights".to_string())?;
    let samples: Vec<Array1<f64>> = (0..m).map(|i| values.row(i).to_owned()).collect();

    let dispersion = |p: ArrayView1<'_, f64>| -> Result<f64, String> {
        let mut acc = 0.0_f64;
        for (i, x) in samples.iter().enumerate() {
            let lg = manifold
                .log_point(p, x.view())
                .map_err(|e| format!("response geometry Fréchet mean log map: {e}"))?;
            let sq = manifold
                .sq_metric_norm(p, lg.view())
                .map_err(|e| format!("response geometry Fréchet mean metric: {e}"))?;
            acc += w[i] * sq;
        }
        Ok(acc)
    };

    const STALL_REL: f64 = 5.0e-3;
    const STALL_PATIENCE: usize = 10;
    const ARMIJO_C1: f64 = 1.0e-4;
    const MAX_BACKTRACK_HALVINGS: usize = 60;
    const ARMIJO_ROUNDOFF_EPS_MULTIPLE: f64 = 8.0;
    // Cap on how many admissible interior starts we descend from before giving
    // up on finding a strictly better basin. Only reached on a positively
    // curved manifold whose data spread makes the mean weakly identified; a
    // small constant keeps the multistart cost `O(1)` in the sample count.
    const MULTISTART_MAX: usize = 8;

    // Safeguarded Riemannian gradient descent on the dispersion `V(P)` from one
    // interior start. Returns the BEST iterate found (least metric-gradient
    // norm), its gradient norm, and its dispersion. It NEVER fails on a budget
    // or stall shortfall: on a positively curved manifold (`St(n,1)=Sⁿ⁻¹`,
    // `Gr(1,n)=ℝPⁿ⁻¹`) a widely spread cloud makes `V` nearly flat, so the
    // descent converges only linearly and can exhaust `max_iter` while still
    // making monotone progress — yet the least-gradient iterate in hand is a
    // perfectly good approximate Fréchet mean and a valid chart origin.
    // Historically the budget-exhausted branch DISCARDED that iterate and
    // returned `Err`, aborting fits that `sphere_frechet_mean` — the SAME
    // manifold reached through a different driver — completes (#2140). All exit
    // conditions (stationarity, stall, rejected step, budget, a moved iterate
    // reaching a sample's cut locus) now agree: keep the best iterate.
    let descend = |start: Array1<f64>| -> (Array1<f64>, f64, f64) {
        let mut p = start;
        let mut f_cur = match dispersion(p.view()) {
            Ok(f) => f,
            // An admissible start has a defined dispersion; guard defensively.
            Err(_) => return (p.clone(), f64::INFINITY, f64::INFINITY),
        };
        let mut best_p = p.clone();
        let mut best_grad = f64::INFINITY;
        let mut best_disp = f_cur;
        let mut stall = 0_usize;
        for _ in 0..max_iter {
            // Riemannian gradient direction ξ = Σ wᵢ log_p(xᵢ) = −½ grad V.
            let mut xi = Array1::<f64>::zeros(ambient);
            let mut log_ok = true;
            for (i, x) in samples.iter().enumerate() {
                match manifold.log_point(p.view(), x.view()) {
                    Ok(lg) => xi.scaled_add(w[i], &lg),
                    // A moved iterate reached a sample's cut locus: stop at the
                    // best iterate held so far rather than aborting the mean.
                    Err(_) => {
                        log_ok = false;
                        break;
                    }
                }
            }
            if !log_ok {
                break;
            }
            let grad_norm = match manifold.sq_metric_norm(p.view(), xi.view()) {
                Ok(g) => g.sqrt(),
                Err(_) => break,
            };
            // Track the best (least-gradient) iterate. `improved` is measured
            // against the PRIOR best so a long linear tail eventually trips the
            // stall guard even while the raw gradient keeps inching down.
            let improved = grad_norm < best_grad * (1.0 - STALL_REL);
            if grad_norm < best_grad {
                best_grad = grad_norm;
                best_p.assign(&p);
                best_disp = f_cur;
            }
            if grad_norm <= tol {
                break;
            }
            if improved {
                stall = 0;
            } else {
                stall += 1;
                if stall >= STALL_PATIENCE {
                    break;
                }
            }

            // Armijo-backtracked unit Karcher step exp_p(t·ξ).
            let pred = grad_norm * grad_norm;
            let f_tol = ARMIJO_ROUNDOFF_EPS_MULTIPLE * f64::EPSILON * (1.0 + f_cur.abs());
            let mut t = 1.0_f64;
            let mut accepted = false;
            for _ in 0..MAX_BACKTRACK_HALVINGS {
                let step = &xi * t;
                let cand = match manifold.exp_point(p.view(), step.view()) {
                    Ok(c) => c,
                    Err(_) => {
                        // The step left the manifold's domain (e.g. a Poincaré
                        // overshoot past the ball boundary); shrink and retry.
                        t *= 0.5;
                        continue;
                    }
                };
                let f_cand = match dispersion(cand.view()) {
                    Ok(f) => f,
                    Err(_) => {
                        t *= 0.5;
                        continue;
                    }
                };
                if f_cand <= f_cur - 2.0 * ARMIJO_C1 * t * pred + f_tol {
                    p = cand;
                    f_cur = f_cand;
                    accepted = true;
                    break;
                }
                t *= 0.5;
            }
            if !accepted {
                break;
            }
        }
        (best_p, best_grad, best_disp)
    };

    // Multistart over admissible interior starts, keeping the mean with the
    // lowest Fréchet dispersion — the generic analogue of `sphere_frechet_mean`.
    //
    // Each seed sample is turned into an interior start by one Karcher averaging
    // step (`exp_seed(Σ wᵢ log_seed xᵢ)`). A fixed seed at `samples[0]` is
    // fragile: if any *other* sample lies at that seed's cut locus the seeding
    // log is undefined even though the mean is well defined — on `Gr(1,n)` two
    // orthogonal lines (principal angle π/2) are exactly such a pair — so we try
    // each sample and skip inadmissible ones.
    //
    // On a Hadamard geometry (`Spd`/`Poincare`) `V` is globally geodesically
    // convex with a UNIQUE minimizer and no cut locus, so the first admissible
    // descent settles the mean (converged → the mean; stalled → its best
    // approximant) and no restart can improve it — one descent, exactly as
    // before. On a positively curved geometry the mean is only LOCALLY unique:
    // a converged descent may sit in a suboptimal basin, so — like
    // `sphere_frechet_mean`, which scores every candidate — we descend up to
    // `MULTISTART_MAX` admissible starts and keep the globally lowest-dispersion
    // one rather than stopping at the first that reaches `tol`. The mean is
    // computed once per fit to set the chart origin, so the bounded extra
    // restarts are amortized.
    let multistart = manifold.frechet_mean_can_be_nonunique();
    let mut best: Option<(Array1<f64>, f64, f64)> = None;
    let mut last_seed_err = String::new();
    let mut descents = 0_usize;
    for seed in &samples {
        let base = match manifold.exp_point(seed.view(), Array1::<f64>::zeros(ambient).view()) {
            Ok(base) => base,
            Err(e) => {
                last_seed_err = e.to_string();
                continue;
            }
        };
        let mut xi = Array1::<f64>::zeros(ambient);
        let mut admissible = true;
        for (i, x) in samples.iter().enumerate() {
            match manifold.log_point(base.view(), x.view()) {
                Ok(lg) => xi.scaled_add(w[i], &lg),
                Err(e) => {
                    last_seed_err = e.to_string();
                    admissible = false;
                    break;
                }
            }
        }
        if !admissible {
            continue;
        }
        let start = match manifold.exp_point(base.view(), xi.view()) {
            Ok(stepped) => stepped,
            Err(e) => {
                last_seed_err = e.to_string();
                continue;
            }
        };

        let (p, grad, disp) = descend(start);
        descents += 1;
        if !multistart {
            // Hadamard: the unique global mean (or its best approximant on a
            // budget shortfall) — a single descent settles it.
            return Ok(p);
        }
        // Positively curved: a converged descent is only a local minimizer, so
        // keep scoring basins by dispersion and return the global best.
        let keep = match &best {
            Some((_, _, best_disp)) => disp < *best_disp,
            None => true,
        };
        if keep {
            best = Some((p, grad, disp));
        }
        if descents >= MULTISTART_MAX {
            break;
        }
    }

    match best {
        Some((p, _, _)) => Ok(p),
        None => Err(format!(
            "response geometry Fréchet mean init: no admissible seed among samples \
             (every sample lies at another's cut locus; last error: {last_seed_err})"
        )),
    }
}

// ── Curvature as an estimand on the response geometry (#944 stage 4 / #1104) ──
//
// `response_geometry="constant_curvature(dim=d)"` does NOT take a fixed κ from
// the user: κ is ESTIMATED from the manifold-valued responses. At each κ the
// family `ConstantCurvature{dim, κ}` is laid down and κ is scored by the HONEST
// change-of-variables likelihood of the observed chart coordinates `yᵢ` w.r.t.
// ambient Lebesgue measure `dy` — the density that is automatically normalised on
// the SAME measure in which the data are observed, regardless of how the manifold
// is parameterised. This is the crux of the #1104 fix.
//
// ## Why dispersion alone (and the self-normalising wrapped Gaussian) is degenerate
//
// The generative model is the wrapped normal `yᵢ = exp_μ(vᵢ)`, `vᵢ` isotropic at
// geodesic scale σ. Its density w.r.t. the Riemannian volume `dvol_κ` is
// `N(sᵢ;0,σ²)/Jᵧ_κ(sᵢ)` with `sᵢ = d_κ(μ,yᵢ)` the geodesic radius and
// `J_κ(s) = (sn_κ(s)/s)^{d−1}` the exp-map volume Jacobian
// (`ConstantCurvature::jacobian_radial`). The naive criterion
// `½nd·ln(Σsᵢ²/nd)` (dispersion only), and even the full `dvol_κ`-density NLL
// `Σ[sᵢ²/2σ² + (d/2)ln2πσ² + ln J_κ(sᵢ)]`, are SCALE-DEGENERATE: rescaling the
// manifold radius `R = 1/√|κ|` rescales every `sᵢ` and every volume element, and
// the σ-profile absorbs the change with no κ information left. That is exactly
// why a `dvol_κ`-normalised (self-normalising) wrapped Gaussian rails, and why an
// intrinsic-volume partition function double-counts: the density is already
// normalised on `dvol_κ`, so re-integrating its volume adds nothing identifying.
//
// ## The restoring force is the ambient (chart) volume element at the DATA points
//
// Curvature is identified only when the abstract manifold is tied to the CONCRETE
// observed chart coordinates `yᵢ`. The data are observed as points of `ℝ^d` under
// Lebesgue `dy`, so the likelihood must be the density w.r.t. `dy`, obtained from
// the `dvol_κ`-density by the chart volume factor `dvol_κ/dy = λ_{yᵢ}^d`,
// `λ_y = 2/(1+κ‖y‖²)`:
//
// ```text
//   −ℓ(κ,μ,σ²) = Σᵢ[ sᵢ²/(2σ²) + (d/2)ln(2πσ²) + ln J_κ(sᵢ) − d·ln λ_{yᵢ} ].
// ```
//
// The new term `−d·Σ ln λ_{yᵢ} = d·Σ ln((1+κ‖yᵢ‖²)/2)` is evaluated at every DATA
// point (not at the mean), so `‖yᵢ‖² > 0` even for mean-centred clouds and it
// supplies a genuine κ-restoring force: it grows like `+d·κ·Σ‖yᵢ‖²` for small κ
// and `→ +∞` as κ→+∞ (each `−ln λ_{yᵢ}→+∞`), exactly opposing the dispersion /
// `ln J_κ` terms which fall as the sphere shrinks. The minimum is therefore
// INTERIOR at the data-generating curvature. None of `ln J_κ` or `λ` depend on σ,
// so σ profiles in closed form `σ̂² = D/(nd)`, `D = Σ sᵢ²`.
//
// ## Reparameterisation invariance / unit-covariance of κ̂
//
// κ carries units of `1/length²`. Under a global rescaling `yᵢ ↦ α·yᵢ` the chart
// of `M_κ` at scale `α` equals the chart of `M_{κ/α²}` at scale 1 (because
// `λ` and every geodesic primitive depend on `y` only through `κ‖y‖²`). The whole
// criterion `V(κ, αy)` therefore equals `V(α²κ, y)`, so its minimiser transforms
// as `κ̂(αy) = κ̂(y)/α²` — the CORRECT covariance of a curvature with units
// `1/length²`. The base point μ is held at the κ-independent flat centroid (NOT
// re-solved per κ): re-solving the Fréchet mean per κ is precisely what
// re-entangles κ with the chart scale and biases the estimate, so it is removed.
//
// `V_p` is a negative log-evidence (lower is better) so κ̂ = argmin V_p; it is the
// full NLL summed over all `n·d` scalar observations, so `2[V_p(0) − V_p(κ̂)]` is
// the Wilks LR statistic with a calibrated χ²₁ flatness reference — exactly the
// contract `profile_ci_walk` / `flatness_lr_test` in `curvature_estimand.rs`
// consume, with no new outer machinery.

/// Outcome of fitting curvature as an estimand on a constant-curvature response
/// geometry: the optimised κ̂, its tangent base point, the profile-likelihood CI,
/// and the interior-point flatness (Wilks) test of κ = 0.
#[derive(Clone, Debug)]
pub struct ResponseCurvatureFit {
    /// The dimension `d` of the constant-curvature response manifold.
    pub dim: usize,
    /// The REML/evidence-optimal curvature κ̂ (argmin of the profiled criterion).
    ///
    /// **Units `1/length²`** — κ̂ is therefore *scale-dependent*: rescaling the
    /// cloud `y ↦ α·y` rescales `κ̂ ↦ κ̂/α²`. For a scale-free statement of how
    /// curved the cloud is, read [`kappa_r2`](Self::kappa_r2) instead. When the
    /// cloud is curved BEYOND what its spread can resolve (it fills a large
    /// fraction of the sphere `S^d(1/√κ̂)`), the optimiser rails to the
    /// chart-resolution cap and [`railed_at_resolution_limit`](Self::railed_at_resolution_limit)
    /// is `true`: κ̂ is then a *lower bound on |κ|*, not a point estimate.
    pub kappa_hat: f64,
    /// The DIMENSIONLESS geometric invariant the cloud actually determines:
    /// `κ̂ · r²` with `r` = [`characteristic_radius`](Self::characteristic_radius).
    /// This is scale-FREE (`κ̂·r²` is invariant under `y ↦ α·y`, since `κ̂ ↦ κ̂/α²`
    /// and `r ↦ α·r`) — the honest answer to "how curved is this cloud relative
    /// to its own spread". `|κ̂·r²| ≪ 1` ⇒ nearly flat at this scale; `κ̂·r² ↗ (π/2)²`
    /// ⇒ the cloud fills the sphere and curvature is at the chart-resolution limit.
    pub kappa_r2: f64,
    /// Characteristic geodesic radius `r` of the cloud at κ = 0 (the doubled-gauge
    /// chart distance `r = 2·max_i‖y_i − μ‖`): the length scale against which κ̂ is
    /// dimensionless. Reported so the caller can convert between scale-dependent κ̂
    /// and the scale-free `κ̂·r²` without re-deriving the chart gauge.
    pub characteristic_radius: f64,
    /// The intrinsic Fréchet-mean base point at κ̂ (the tangent expansion point
    /// the scalar GAMs are fitted around).
    pub base: Array1<f64>,
    /// Profiled criterion value `V_p(κ̂)` (concentrated negative log-evidence).
    pub v_p_hat: f64,
    /// `true` when the κ̂ search converged ONTO the chart-resolution cap rather
    /// than an interior optimum: the data want curvature at or beyond the
    /// conjugate radius of their geodesic spread (the cloud fills the sphere).
    /// In that case κ̂ / the CI upper end are NOT a resolved point estimate but a
    /// HONEST "curvature exceeds chart-resolvable range at this scale" flag — the
    /// caller must report it as such and never as a silent `κ̂ = ci_hi`. The
    /// hyperbolic side cannot rail this way (κ < 0 has no conjugate radius), so a
    /// rail here always means strongly spherical relative to the spread.
    pub railed_at_resolution_limit: bool,
    /// `true` only when the SIGN of κ̂ is statistically resolved — i.e. the
    /// profile-likelihood CI excludes 0 (`profile_ci.verdict ≠ Flat`).
    ///
    /// ## Why a point estimate alone is not enough (the #944/#1059 flat-floor)
    ///
    /// Curvature is resolvable only through the dimensionless product `κ·r²`
    /// (see [`kappa_r2`](Self::kappa_r2)); the per-point Fisher information for κ
    /// scales like `σ⁴`. When the cloud is nearly flat at its own scale
    /// (`|κ·r²| ≪ 1`), the profiled criterion is so shallow that its single-cloud
    /// argmin κ̂ can land on the WRONG SIDE OF ZERO purely by Monte-Carlo
    /// fluctuation — empirically a coin-flip below `|κ·r²| ≈ 0.03`, reliable above
    /// `≈ 0.09` (the #944 power curve). The estimand itself is UNBIASED (the
    /// criterion averaged over clouds minimises exactly at κ⋆), so this is a
    /// resolution limit, not a bias.
    ///
    /// The CI, in contrast, is honest in this regime: at an under-resolved
    /// operating point it reports `Flat` (straddles 0) rather than a confident
    /// wrong sign — it essentially never claims the wrong-signed geometry. So the
    /// SIGN-bearing summary the caller may quote is the CI verdict, not the bare
    /// κ̂. This flag exposes that contract on the point-estimate surface: when it
    /// is `false`, κ̂'s sign is noise — the caller must report "curvature not
    /// resolved at this scale (|κ·r²| too small)" and quote the CI / `kappa_r2`,
    /// never a sign-confident κ̂. It is the flat-floor twin of
    /// [`railed_at_resolution_limit`](Self::railed_at_resolution_limit) (the
    /// spherical-cap rail); together they bracket the two ends of the resolvable
    /// `κ·r²` band where κ̂ is a genuine interior point estimate.
    pub sign_resolved: bool,
    /// Profile-likelihood CI for κ and the geometry verdict from its sign.
    pub profile_ci: crate::curvature_estimand::KappaProfileCi,
    /// Interior-point χ²₁ likelihood-ratio test of flatness (κ = 0).
    pub flatness: crate::curvature_estimand::FlatnessTest,
}

/// Chart-validity bounds on κ for a constant-curvature response geometry built
/// from the supplied responses, plus the characteristic geodesic radius
/// `ρ_max = 2·max_i‖y_i − μ‖` against which κ is made dimensionless.
///
/// Returns `(kappa_min, kappa_max, rho_max)`.
///
/// * **Lower (hyperbolic) bound.** The κ-stereographic chart requires
///   `1 + κ‖x‖² > 0` at every point measured from the chart origin, i.e.
///   `κ > −1/R²` with `R² = max_i ‖y_i‖²`. With a safety margin: `−0.999/R²`.
/// * **Upper (spherical) bound.** Unlike the hyperbolic side this is NOT
///   unbounded: on a sphere of curvature κ the geodesic radius cannot exceed the
///   conjugate radius `π/√κ`, beyond which the exp-map volume Jacobian
///   `J_κ = (sn_κ/·)^{d−1}` changes sign (clamped to 0 here) and `ln J_κ` would
///   collapse `V_p` toward `−∞`, railing the optimiser onto a spurious shell.
///   The κ = 0 geodesic radius of the farthest point from the centroid is
///   `ρ_max = 2·max_i‖y_i − μ‖` (doubled-gauge chart). We cap κ so that radius
///   stays strictly inside the first conjugate shell with a 10% margin:
///   `√κ·ρ_max ≤ 0.9π ⇒ κ_max = (0.9π / ρ_max)²`. This keeps every geodesic
///   radius before the antipodal singularity along the whole search/CI walk.
///
/// `κ_max` is the chart-RESOLUTION limit of the cloud: at it the geodesic spread
/// fills `(0.9π)² ≈ (π/2·1.8)²` of the conjugate shell, i.e. the cloud nearly
/// fills the sphere `S^d(1/√κ_max)`. The DIMENSIONLESS product `κ_max·ρ_max²
/// = (0.9π)²` is fixed and data-scale-free — it is the natural "the cloud is
/// maximally curved relative to its spread" sentinel the rail check compares κ̂ to.
fn response_kappa_bounds(values: ArrayView2<'_, f64>) -> (f64, f64, f64) {
    let (n_rows, dim) = values.dim();
    // ‖y_i‖² from the chart origin (governs the λ / hyperbolic-chart constraint).
    let mut r2_max = 0.0_f64;
    for row in values.outer_iter() {
        let r2 = row.dot(&row);
        if r2 > r2_max {
            r2_max = r2;
        }
    }
    // ‖y_i − μ‖² from the centroid (governs the spherical conjugate-radius cap).
    let mut centroid = Array1::<f64>::zeros(dim.max(1));
    if n_rows > 0 && dim > 0 {
        for row in values.outer_iter() {
            centroid += &row;
        }
        centroid.mapv_inplace(|v| v / n_rows as f64);
    }
    let mut s2_max = 0.0_f64;
    if dim > 0 {
        for row in values.outer_iter() {
            let diff = &row - &centroid;
            let r2 = diff.dot(&diff);
            if r2 > s2_max {
                s2_max = r2;
            }
        }
    }
    if r2_max <= 0.0 && s2_max <= 0.0 {
        // Degenerate (all points at the origin): κ is unidentified; use a wide
        // symmetric default so the optimiser/CI report a flat, unbounded result.
        return (-1.0e6, 1.0e6, 0.0);
    }
    // Keep a safety margin off the singular hyperbolic boundary.
    let kappa_min = if r2_max > 0.0 {
        -0.999 / r2_max
    } else {
        -1.0e6
    };
    // Conjugate-radius cap: ρ_max = 2·max‖y_i − μ‖ is the κ=0 geodesic radius.
    let rho_max = 2.0 * s2_max.sqrt();
    let kappa_max = if s2_max > 0.0 {
        let edge = 0.9 * std::f64::consts::PI / rho_max;
        edge * edge
    } else {
        1.0e6
    };
    (kappa_min, kappa_max, rho_max)
}

/// Profiled curvature criterion `V_p(κ)` for the constant-curvature response
/// geometry: the σ-profiled HONEST change-of-variables negative log-likelihood of
/// the observed chart coordinates `y_i` at curvature `κ`, expressed w.r.t. ambient
/// Lebesgue measure `dy`. Lower is better (κ̂ = argmin). Returns `(V_p, base)`;
/// the base point is the κ-INDEPENDENT flat centroid (the tangent expansion point
/// that the scalar GAMs are fitted around), held fixed across κ so the estimate is
/// not re-entangled with the chart scale.
///
/// The model is the wrapped normal `y_i = exp_{μ,κ}(v_i)` with isotropic geodesic
/// scale σ; `s_i = d_κ(μ, y_i)` is the geodesic radius and `J_κ(s)` the exp-map
/// volume Jacobian. The density on the Riemannian volume `dvol_κ` is
/// `N(s_i;0,σ²)/J_κ(s_i)`; converting to ambient `dy` multiplies by the chart
/// volume factor `λ_{y_i}^d`, `λ_y = 2/(1+κ‖y‖²)`. The negative log-likelihood is
///
/// ```text
///   −ℓ(κ,σ²) = Σ_i[ s_i²/(2σ²) + (d/2)ln(2πσ²) + ln J_κ(s_i) − d·ln λ_{y_i} ].
/// ```
///
/// `ln J_κ` and `λ` do not depend on σ, so σ profiles in closed form
/// `σ̂² = D/(nd)`, `D = Σ s_i²`. The `−d·Σ ln λ_{y_i}` term — evaluated at the DATA
/// points, not the mean — is the κ-restoring force that breaks the scale
/// degeneracy of the dispersion / `dvol_κ`-density alone (see the module notes).
/// Additive constants independent of κ are kept implicit; they cancel in every
/// LR / profile-drop the CI machinery forms. μ is the closed-form flat centroid,
/// so the criterion is a pure function of κ with no inner tolerance/iteration
/// budget (the outer κ̂ search owns those).
pub fn response_curvature_criterion(
    values: ArrayView2<'_, f64>,
    dim: usize,
    kappa: f64,
) -> Result<(f64, Array1<f64>), String> {
    if !kappa.is_finite() {
        return Err("response curvature criterion: kappa must be finite".into());
    }
    let (n_rows, cols) = values.dim();
    if n_rows == 0 || cols != dim || dim == 0 {
        return Err(format!(
            "response curvature criterion: values must be N×{dim} with N >= 1"
        ));
    }
    // κ-independent base point: the flat (ambient) centroid. Holding μ fixed across
    // κ is the de-entangling move — re-solving the Fréchet mean per κ couples the
    // base to the chart scale and biases κ̂ (#1104 root cause).
    let mut base = Array1::<f64>::zeros(dim);
    for row in values.outer_iter() {
        base += &row;
    }
    base.mapv_inplace(|v| v / n_rows as f64);

    let chart = ConstantCurvature::new(dim, kappa);
    // Reject κ at/over the chart boundary (1 + κ‖x‖² ≤ 0) at the centroid or any
    // data point: the geodesic primitives are undefined there. The bracket in
    // `response_kappa_bounds` keeps the optimiser strictly inside, but a CI/LR
    // probe can still land on the edge, so guard rather than panic.
    chart
        .conformal_factor(base.view())
        .map_err(|e| format!("response curvature criterion: base off chart: {e}"))?;

    let d = dim as f64;
    let mut dispersion = 0.0_f64; // D = Σ s_i²
    let mut ln_jac = 0.0_f64; // Σ ln J_κ(s_i)
    let mut ln_lambda = 0.0_f64; // Σ ln λ_{y_i}
    // Geodesic radii s_i = d_κ(μ, y_i) for every row, computed in a single
    // batched pass (four rows per SIMD lane-group). `distance_batch` is
    // bit-for-bit identical to the per-row `distance`, so D, Σ ln J, and Σ ln λ
    // below are unchanged; it also validates each y_i is in-chart.
    let mut radii = vec![0.0_f64; n_rows];
    chart
        .distance_batch(base.view(), values, &mut radii)
        .map_err(|e| format!("response curvature criterion distance: {e}"))?;
    for (row, &s) in values.outer_iter().zip(radii.iter()) {
        dispersion += s * s;
        // ln J_κ(s_i): exp-map volume Jacobian (≥ 0); floor before the log so the
        // conjugate-shell clamp (J → 0 on the κ>0 antipodal shell) is a large
        // finite penalty rather than −∞.
        ln_jac += chart.jacobian_radial(s).max(1.0e-300).ln();
        // ln λ_{y_i} = ln(2) − ln(1 + κ‖y_i‖²); `conformal_factor` validates chart.
        let lam = chart
            .conformal_factor(row)
            .map_err(|e| format!("response curvature criterion conformal factor: {e}"))?;
        ln_lambda += lam.ln();
    }
    let nobs = (n_rows * dim) as f64;
    // Floor the dispersion so a (near-)perfect flat fit does not blow ln up; the
    // floor is far below any genuine residual scale and cancels in profile drops.
    let disp = dispersion.max(1.0e-300 * nobs.max(1.0));

    // σ profiles in closed form: σ̂² = D/(nd). Substituting and dropping the
    // κ-independent constant (nd/2)(1 + ln 2π):
    //   V_p(κ) = (nd/2)·ln(D/(nd)) + Σ ln J_κ(s_i) − d·Σ ln λ_{y_i}.
    let v_p = 0.5 * nobs * (disp / nobs).ln() + ln_jac - d * ln_lambda;
    Ok((v_p, base))
}

/// Fit curvature as an estimand on a constant-curvature response geometry.
///
/// κ̂ is the minimiser of the profiled criterion [`response_curvature_criterion`]
/// (the σ-profiled honest change-of-variables negative log-evidence of the wrapped
/// normal w.r.t. ambient measure), found by a golden-section search inside the
/// chart-validity bracket. The base point μ is the κ-independent flat centroid, so
/// every `V_p` evaluation scores the SAME geometry without re-entangling κ with the
/// chart scale (the #1104 fix). The exact outer
/// curvature `V_p''(κ̂)` is taken by a central second difference of the same
/// criterion and handed to [`profile_ci_walk`](crate::profile_ci_walk)
/// to size the initial Wald step; the CI itself is the exact χ²₁ profile crossing.
/// Flatness is the interior-point χ²₁ LR test
/// [`flatness_lr_test`](crate::flatness_lr_test). κ = 0 is an interior
/// point of the analytic `S^d ← ℝ^d → H^d` family, so no boundary correction is
/// applied. Returns the κ̂, its tangent base point, the profile CI, and the Wilks
/// flatness test for the fit summary.
///
/// ## Scale-awareness and honest railing (#1104)
///
/// κ has units `1/length²`, so a cloud of characteristic geodesic radius `r`
/// resolves only the DIMENSIONLESS product `κ·r²` (every chart primitive depends
/// on `y` through `κ‖y‖²`, hence `V(κ, αy) = V(α²κ, y)` and `κ̂ ↦ κ̂/α²` under
/// `y ↦ αy`). The fit therefore also returns:
/// * `kappa_r2 = κ̂·r²` — the scale-FREE invariant the cloud actually determines
///   (how curved relative to its own spread), and `characteristic_radius = r`;
/// * `railed_at_resolution_limit` — `true` when the data want curvature at or
///   beyond the conjugate radius of their spread (the cloud fills the sphere),
///   so the search converges onto the spherical cap. There κ̂ is a LOWER BOUND on
///   `|κ|`, not a resolved point estimate, and the caller must report "curvature
///   exceeds chart-resolvable range at this scale" rather than silently quoting
///   `κ̂ = ci_hi`. This is the #1104 fix: a tightly-concentrated near-spherical
///   cloud (e.g. unit-normalised OLMo activations) no longer SILENTLY rails to a
///   huge scale-dependent `ci_hi` while claiming a point estimate + CI.
pub fn fit_response_curvature(
    values: ArrayView2<'_, f64>,
    dim: usize,
    level: f64,
    tol: f64,
    max_iter: usize,
) -> Result<ResponseCurvatureFit, String> {
    if dim == 0 {
        return Err("constant-curvature response geometry requires dim >= 1".into());
    }
    let (n_rows, cols) = values.dim();
    if n_rows == 0 || cols != dim {
        return Err(format!(
            "constant-curvature response geometry: values must be N×{dim} with N >= 1"
        ));
    }
    if !(level > 0.0 && level < 1.0) {
        return Err("response curvature CI level must lie in (0, 1)".into());
    }
    let (kappa_min, kappa_max, rho_max) = response_kappa_bounds(values);

    // `V_p` as a closure over the criterion; threaded through both the κ̂ search
    // and the CI walk. Every evaluation uses the same κ-independent flat-centroid
    // base, so the criterion is a clean 1-D function of κ.
    let mut v_p = |kappa: f64| -> Result<f64, String> {
        response_curvature_criterion(values, dim, kappa).map(|(v, _)| v)
    };

    // ── κ̂: golden-section minimisation inside the chart bracket. ────────────
    // The dispersion criterion is smooth and unimodal in practice; golden
    // section is derivative-free and respects the bracket bounds exactly.
    const GOLDEN_INV: f64 = 0.618_033_988_749_894_8; // 1/φ
    let mut a = kappa_min;
    let mut b = kappa_max;
    let mut c = b - GOLDEN_INV * (b - a);
    let mut d_pt = a + GOLDEN_INV * (b - a);
    let mut fc = v_p(c)?;
    let mut fd = v_p(d_pt)?;
    let ktol = (tol * (kappa_max - kappa_min)).max(tol).max(1.0e-12);
    for _ in 0..max_iter {
        if (b - a).abs() <= ktol {
            break;
        }
        if fc < fd {
            b = d_pt;
            d_pt = c;
            fd = fc;
            c = b - GOLDEN_INV * (b - a);
            fc = v_p(c)?;
        } else {
            a = c;
            c = d_pt;
            fc = fd;
            d_pt = a + GOLDEN_INV * (b - a);
            fd = v_p(d_pt)?;
        }
    }
    let kappa_hat = 0.5 * (a + b);
    let (v_p_hat, base) = response_curvature_criterion(values, dim, kappa_hat)?;

    // ── Honest chart-resolution-rail detection. ─────────────────────────────
    // The spherical cap κ_max is the curvature at which the cloud's geodesic
    // spread ρ_max fills `(0.9π)²` of the conjugate shell — i.e. the cloud nearly
    // fills the sphere S^d(1/√κ_max). When the criterion's optimum sits AT that
    // cap (the data want κ ≥ κ_max, but the chart cannot resolve a sphere smaller
    // than the cloud), the search converges onto the upper bracket and κ̂ ≈ κ_max
    // is NOT a resolved point estimate — it is a lower bound on |κ|. We flag this
    // so the caller reports "curvature exceeds chart-resolvable range at this
    // scale" instead of silently quoting κ̂ / ci_hi as if interior. The detection
    // is scale-free: it triggers when κ̂ lands within the final golden-section
    // resolution of κ_max (the dimensionless product κ̂·ρ_max² ↗ (0.9π)²), never
    // by an absolute κ threshold. The hyperbolic side has no conjugate radius, so
    // only the spherical (upper) cap can rail this way.
    let span = kappa_max - kappa_min;
    let rail_margin = (0.02 * span).max(ktol);
    let railed_at_resolution_limit = kappa_hat >= kappa_max - rail_margin;

    // Dimensionless scale-free invariant κ̂·r²: the geometric content the cloud
    // actually determines (invariant under y ↦ αy). r = ρ_max is the κ=0 doubled-
    // gauge characteristic radius; for a degenerate (point) cloud r = 0 and the
    // product is 0 (κ unidentified). This is what the caller should report as the
    // honest "how curved relative to its spread" number alongside the dimensional κ̂.
    let kappa_r2 = kappa_hat * rho_max * rho_max;

    // Exact outer curvature V_p''(κ̂) by a central second difference, on a step
    // scaled to the bracket; only used to size the Wald bracket of the CI walk.
    let h = (1.0e-3 * (kappa_max - kappa_min)).max(1.0e-6);
    let v_pp = if (kappa_hat - h) > kappa_min && (kappa_hat + h) < kappa_max {
        let vp = v_p(kappa_hat + h)?;
        let vm = v_p(kappa_hat - h)?;
        (vp - 2.0 * v_p_hat + vm) / (h * h)
    } else {
        // Near a bound: leave it to the walk's default step.
        f64::NAN
    };

    let profile_ci = crate::curvature_estimand::profile_ci_walk(
        &mut v_p, kappa_hat, v_pp, kappa_min, kappa_max, level, ktol,
    )?;
    let flatness = crate::curvature_estimand::flatness_lr_test(&mut v_p, kappa_hat)?;

    // The sign of κ̂ is statistically resolved iff the profile CI excludes 0 — the
    // CI is the honest sign-bearing summary (it reports Flat under-resolution rather
    // than a confident wrong sign), so we mirror its verdict onto the point-estimate
    // surface. Below the resolvable `κ·r²` floor (`|κ·r²| ≪ 1`) the bare κ̂ argmin can
    // flip sign on Monte-Carlo noise, so `false` here means "do not quote κ̂'s sign".
    let sign_resolved = !matches!(
        profile_ci.verdict,
        crate::curvature_estimand::CurvatureVerdict::Flat
    );

    Ok(ResponseCurvatureFit {
        dim,
        kappa_hat,
        kappa_r2,
        characteristic_radius: rho_max,
        railed_at_resolution_limit,
        sign_resolved,
        base,
        v_p_hat,
        profile_ci,
        flatness,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::{Array2, array};

    fn round_trip(manifold: ResponseManifold, values: Array2<f64>) {
        let base =
            response_frechet_mean(manifold, values.view(), None, 1e-12, 500).expect("frechet mean");
        let tangent = response_log_map(manifold, values.view(), base.view()).expect("log map");
        let back = response_exp_map(manifold, tangent.view(), base.view()).expect("exp map");
        for row in 0..values.nrows() {
            for col in 0..values.ncols() {
                assert!(
                    (back[[row, col]] - values[[row, col]]).abs() < 1e-6,
                    "{manifold:?} exp∘log mismatch at ({row},{col}): {} vs {}",
                    back[[row, col]],
                    values[[row, col]]
                );
            }
        }
    }

    #[test]
    fn spd_round_trip_and_mean() {
        // Three 2×2 SPD matrices, row-major flat.
        let values = array![
            [2.0, 0.0, 0.0, 1.0],
            [1.0, 0.3, 0.3, 2.0],
            [3.0, -0.5, -0.5, 1.5],
        ];
        round_trip(ResponseManifold::Spd { n: 2 }, values);
    }

    #[test]
    fn grassmann_round_trip_and_mean() {
        // Gr(1, 3): unit columns (lines through the origin), n·k = 3 flat.
        let values = array![[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.6, 0.8, 0.0],];
        round_trip(ResponseManifold::Grassmann { k: 1, n: 3 }, values);
    }

    #[test]
    fn stiefel_round_trip_and_mean() {
        // St(1, 3): unit 1-frames in ℝ³ (== sphere S²).
        let values = array![[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.6, 0.8],];
        round_trip(ResponseManifold::Stiefel { k: 1, n: 3 }, values);
    }

    #[test]
    fn stiefel_k2_round_trip_and_mean_n_lt_2k() {
        // St(3, 2): three orthonormal 2-frames in ℝ³ clustered near [e0, e1],
        // exercising the genuine canonical-metric logarithm (k ≥ 2) through the
        // full Karcher-mean → log → exp round trip. This is the n < 2k regime
        // (n = 3 < 2k = 4) where the economical 2k-block form is rank-deficient.
        // Before the k ≥ 2 Stiefel logarithm existed this aborted in
        // Fréchet-mean init with a misleading cut-locus error (#1637).
        let (c2, s2) = (0.2_f64.cos(), 0.2_f64.sin());
        let (c1, s1) = (0.15_f64.cos(), 0.15_f64.sin());
        let values = array![
            [1.0, 0.0, 0.0, 1.0, 0.0, 0.0],
            [c2, 0.0, 0.0, 1.0, s2, 0.0],
            [1.0, 0.0, 0.0, c1, 0.0, s1],
        ];
        round_trip(ResponseManifold::Stiefel { k: 2, n: 3 }, values);
    }

    #[test]
    fn stiefel_k2_round_trip_and_mean_n_ge_2k() {
        // St(4, 2): the n ≥ 2k regime (n = 4 = 2k), clustered 2-frames in ℝ⁴.
        let (c0, s0) = (0.1_f64.cos(), 0.1_f64.sin());
        let (c1, s1) = (0.12_f64.cos(), 0.12_f64.sin());
        let values = array![
            [1.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0],
            [c0, 0.0, 0.0, 1.0, s0, 0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0, c1, 0.0, 0.0, 0.0, s1],
        ];
        round_trip(ResponseManifold::Stiefel { k: 2, n: 4 }, values);
    }

    #[test]
    fn poincare_round_trip_and_mean() {
        let values = array![[0.1, 0.2], [-0.3, 0.1], [0.2, -0.25],];
        round_trip(
            ResponseManifold::Poincare {
                dim: 2,
                curvature: -1.0,
            },
            values,
        );
    }

    /// Deterministic Fibonacci-lattice cover of S² (== `St(3,1)` == `Gr(1,3)`
    /// projectively), spread over the WHOLE sphere. This is the widely spread
    /// cloud that makes the Fréchet objective nearly flat, so a single-seed
    /// Karcher descent converges only linearly and exhausts a `max_iter=256`
    /// budget — the #2140 trigger.
    fn fibonacci_sphere(n: usize) -> Array2<f64> {
        let mut v = Array2::<f64>::zeros((n, 3));
        let golden = std::f64::consts::PI * (1.0 + 5.0_f64.sqrt());
        for idx in 0..n {
            let i = idx as f64 + 0.5;
            let phi = (1.0 - 2.0 * i / n as f64).acos();
            let theta = golden * i;
            v[[idx, 0]] = theta.cos() * phi.sin();
            v[[idx, 1]] = theta.sin() * phi.sin();
            v[[idx, 2]] = phi.cos();
        }
        v
    }

    /// Uniform-weight Fréchet objective `(1/M) Σ ‖log_p Xᵢ‖²_p` — the dispersion
    /// the driver minimizes. Lower is a better mean. Uses the module-private
    /// per-point primitives directly (available inside this child module).
    fn frechet_objective(
        manifold: ResponseManifold,
        values: ArrayView2<'_, f64>,
        p: ArrayView1<'_, f64>,
    ) -> f64 {
        let mut acc = 0.0;
        for row in 0..values.nrows() {
            let lg = manifold.log_point(p, values.row(row)).expect("log map");
            acc += manifold
                .sq_metric_norm(p, lg.view())
                .expect("metric norm");
        }
        acc / values.nrows() as f64
    }

    #[test]
    fn stiefel_k1_frechet_mean_accepts_widely_spread_sphere_cloud() {
        // #2140: `St(3,1)` IS S²; a widely spread cloud makes the single-seed
        // generic Karcher descent exhaust its `max_iter=256` budget while still
        // inching downhill. The driver used to DISCARD the best iterate and
        // return `Err`, aborting the whole fit — even though `response_geometry
        // ="sphere"` on the identical data succeeds. It must instead return the
        // best (near-stationary) on-manifold mean.
        let values = fibonacci_sphere(80);
        let manifold = ResponseManifold::Stiefel { k: 1, n: 3 };
        let mean = response_frechet_mean(manifold, values.view(), None, 1.0e-12, 256)
            .expect("widely spread sphere cloud must yield a Fréchet mean, not a budget error");

        // On-manifold: a genuine unit vector (a valid chart origin the tangent
        // regression can expand around).
        assert_eq!(mean.len(), 3);
        let nrm = (mean[0] * mean[0] + mean[1] * mean[1] + mean[2] * mean[2]).sqrt();
        assert!((nrm - 1.0).abs() < 1e-9, "mean must be unit-norm, got {nrm}");
        assert!(mean.iter().all(|c| c.is_finite()), "mean must be finite");

        // And it is a genuine minimizer, not just any on-manifold point: its
        // dispersion beats an arbitrary data point's.
        let obj_mean = frechet_objective(manifold, values.view(), mean.view());
        let obj_pt0 = frechet_objective(manifold, values.view(), values.row(0));
        assert!(
            obj_mean < obj_pt0,
            "Fréchet mean dispersion {obj_mean} must beat a data point's {obj_pt0}"
        );
    }

    #[test]
    fn stiefel_k1_and_sphere_drivers_agree_on_widely_spread_cloud() {
        // `stiefel(k=1)` (generic driver) and `sphere` (dedicated driver) are
        // the SAME manifold; on the identical widely spread cloud the generic
        // driver's mean must be essentially as good as the sphere driver's —
        // proof the returned budget-exhausted iterate is a true near-mean, not a
        // degraded surrogate.
        let values = fibonacci_sphere(80);
        let stiefel = ResponseManifold::Stiefel { k: 1, n: 3 };
        let generic = response_frechet_mean(stiefel, values.view(), None, 1.0e-12, 256)
            .expect("generic driver must return a mean");
        let sphere = crate::manifolds::sphere::sphere_frechet_mean(values.view(), None, 1.0e-12, 256)
            .expect("sphere driver must return a mean");
        let sphere = Array1::from(sphere);

        let obj_generic = frechet_objective(stiefel, values.view(), generic.view());
        let obj_sphere = frechet_objective(stiefel, values.view(), sphere.view());
        assert!(
            obj_generic <= obj_sphere + 1e-3,
            "generic-driver dispersion {obj_generic} must match the sphere driver's {obj_sphere}"
        );
    }

    #[test]
    fn budget_exhausted_generic_frechet_returns_best_iterate_not_error() {
        // Directly pin the fixed branch: a `max_iter` too small to ever reach
        // stationarity must STILL return the best on-manifold iterate (matching
        // the stall and rejected-step exits), never the historical
        // "did not reach stationarity within max_iter" error.
        let values = fibonacci_sphere(60);
        for manifold in [
            ResponseManifold::Stiefel { k: 1, n: 3 },
            ResponseManifold::Grassmann { k: 1, n: 3 },
        ] {
            let mean = response_frechet_mean(manifold, values.view(), None, 1.0e-12, 1)
                .unwrap_or_else(|e| panic!("{manifold:?} budget=1 must return best iterate: {e}"));
            assert!(
                mean.iter().all(|c| c.is_finite()),
                "{manifold:?} best iterate must be finite"
            );
            let nrm = mean.iter().map(|c| c * c).sum::<f64>().sqrt();
            assert!(
                (nrm - 1.0).abs() < 1e-9,
                "{manifold:?} best iterate must stay on-manifold (unit-norm), got {nrm}"
            );
        }
    }

    #[test]
    fn hadamard_frechet_mean_uses_single_descent_fast_path() {
        // Spd/Poincaré are Hadamard: the mean is unique, so `multistart` must be
        // OFF and a well-clustered cloud converges to `grad ≤ tol` on the first
        // seed. Guard the classifier and the fast path together.
        assert!(!ResponseManifold::Spd { n: 2 }.frechet_mean_can_be_nonunique());
        assert!(
            !ResponseManifold::Poincare {
                dim: 2,
                curvature: -1.0
            }
            .frechet_mean_can_be_nonunique()
        );
        assert!(ResponseManifold::Stiefel { k: 1, n: 3 }.frechet_mean_can_be_nonunique());
        assert!(ResponseManifold::Grassmann { k: 1, n: 4 }.frechet_mean_can_be_nonunique());
        assert!(
            ResponseManifold::ConstantCurvature {
                dim: 2,
                kappa: 3.0
            }
            .frechet_mean_can_be_nonunique()
        );
        assert!(
            !ResponseManifold::ConstantCurvature {
                dim: 2,
                kappa: -3.0
            }
            .frechet_mean_can_be_nonunique()
        );

        // A tight SPD cluster still converges to the unique Karcher mean.
        let values = array![
            [2.0, 0.0, 0.0, 1.0],
            [2.1, 0.05, 0.05, 1.02],
            [1.95, -0.03, -0.03, 0.98],
        ];
        let mean = response_frechet_mean(ResponseManifold::Spd { n: 2 }, values.view(), None, 1e-12, 500)
            .expect("SPD cluster must converge");
        assert!(mean.iter().all(|c| c.is_finite()));
    }

    #[test]
    fn resolver_rejects_bad_shapes() {
        assert!(ResponseManifold::resolve("grassmann", Some(2), Some(3), None, None).is_err());
        assert!(ResponseManifold::resolve("spd", None, None, None, None).is_err());
        assert!(ResponseManifold::resolve("poincare", None, None, Some(2), Some(1.0)).is_err());
        assert!(ResponseManifold::resolve("nonsense", None, None, None, None).is_err());
        assert_eq!(
            ResponseManifold::resolve("spd", Some(3), None, None, None).unwrap(),
            ResponseManifold::Spd { n: 3 }
        );
    }

    #[test]
    fn parse_infers_shapes_from_columns() {
        // SPD: n from the perfect-square column count.
        assert_eq!(
            ResponseManifold::parse("spd", 9).unwrap(),
            ResponseManifold::Spd { n: 3 }
        );
        assert!(ResponseManifold::parse("spd", 8).is_err());
        // Grassmann/Stiefel: n inferred as cols / k.
        assert_eq!(
            ResponseManifold::parse("grassmann(k=2)", 10).unwrap(),
            ResponseManifold::Grassmann { k: 2, n: 5 }
        );
        assert_eq!(
            ResponseManifold::parse("Stiefel( k = 2 , n = 4 )", 8).unwrap(),
            ResponseManifold::Stiefel { k: 2, n: 4 }
        );
        assert!(ResponseManifold::parse("grassmann", 10).is_err());
        assert!(ResponseManifold::parse("grassmann(k=3)", 10).is_err());
        // Poincaré: dim = cols, default curvature -1.
        assert_eq!(
            ResponseManifold::parse("poincare", 3).unwrap(),
            ResponseManifold::Poincare {
                dim: 3,
                curvature: -1.0
            }
        );
        assert_eq!(
            ResponseManifold::parse("poincare(curvature=-0.5)", 3).unwrap(),
            ResponseManifold::Poincare {
                dim: 3,
                curvature: -0.5
            }
        );
        assert!(ResponseManifold::parse("hyperbolic", 3).is_err());
    }

    #[test]
    fn dispatch_round_trips_through_user_label() {
        // Drive the full string-selected user path for each geometry: parse the
        // label, build the intrinsic base, log to the tangent, exp back.
        let cases: Vec<(&str, Array2<f64>)> = vec![
            (
                "spd",
                array![
                    [2.0, 0.0, 0.0, 1.0],
                    [1.0, 0.3, 0.3, 2.0],
                    [3.0, -0.5, -0.5, 1.5],
                ],
            ),
            (
                "grassmann(k=1)",
                array![[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.6, 0.8, 0.0]],
            ),
            (
                "stiefel(k=1)",
                array![[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.6, 0.8]],
            ),
            ("poincare", array![[0.1, 0.2], [-0.3, 0.1], [0.2, -0.25]]),
        ];
        for (label, values) in cases {
            let (tangent, base, canonical) =
                dispatch_log_map(values.view(), label, None, None).expect("dispatch log");
            assert!(canonical.starts_with(label.split('(').next().unwrap()));
            let back = dispatch_exp_map(tangent.view(), label, base.view()).expect("dispatch exp");
            for row in 0..values.nrows() {
                for col in 0..values.ncols() {
                    assert!(
                        (back[[row, col]] - values[[row, col]]).abs() < 1e-6,
                        "{label} exp∘log mismatch at ({row},{col}): {} vs {}",
                        back[[row, col]],
                        values[[row, col]]
                    );
                }
            }
        }
    }

    #[test]
    fn ambient_dim_matches_layout() {
        assert_eq!(ResponseManifold::Spd { n: 3 }.ambient_dim(), 9);
        assert_eq!(ResponseManifold::Grassmann { k: 2, n: 5 }.ambient_dim(), 10);
        assert_eq!(ResponseManifold::Stiefel { k: 2, n: 4 }.ambient_dim(), 8);
        assert_eq!(
            ResponseManifold::Poincare {
                dim: 4,
                curvature: -1.0
            }
            .ambient_dim(),
            4
        );
    }

    /// #2125: a weighted response-geometry fit must linearize around the
    /// *weighted* Fréchet mean. `dispatch_log_map` picks the tangent base point;
    /// before the fix it hard-passed `None` for the weights, so the chart origin
    /// was the unweighted intrinsic mean even when the tangent regression was
    /// weighted — a biased linearization. Here Stiefel(k=1,n=3) is the sphere S²:
    /// two clusters of unit vectors 90° apart, with weights concentrated on the
    /// first cluster, must move the base point toward that cluster.
    #[test]
    fn dispatch_log_map_uses_weighted_frechet_mean() {
        let a = 0.15_f64;
        // Two clusters on the great circle z = 0: cluster A about [1,0,0]
        // (rows 0,1) and cluster B about [0,1,0] (rows 2,3). Every row is an
        // exact unit vector (cos²+sin²=1), so Stiefel(k=1) accepts them.
        let values = array![
            [a.cos(), a.sin(), 0.0],
            [(-a).cos(), (-a).sin(), 0.0],
            [(std::f64::consts::FRAC_PI_2 - a).cos(), (std::f64::consts::FRAC_PI_2 - a).sin(), 0.0],
            [(std::f64::consts::FRAC_PI_2 + a).cos(), (std::f64::consts::FRAC_PI_2 + a).sin(), 0.0],
        ];
        // Heavily weight cluster A: the weighted mean must sit near [1,0,0],
        // whereas the unweighted mean sits near the 45° bisector.
        let weights = array![50.0_f64, 50.0, 1.0, 1.0];
        let manifold = ResponseManifold::Stiefel { k: 1, n: 3 };

        let geodesic = |u: ArrayView1<'_, f64>, v: ArrayView1<'_, f64>| -> f64 {
            u.dot(&v).clamp(-1.0, 1.0).acos()
        };

        let unweighted_ref =
            response_frechet_mean(manifold, values.view(), None, 1e-12, 256).expect("unweighted");
        let weighted_ref =
            response_frechet_mean(manifold, values.view(), Some(weights.view()), 1e-12, 256)
                .expect("weighted");
        // Sanity: the two intrinsic means genuinely differ, so this design can
        // distinguish a weighted from an unweighted base point.
        assert!(
            geodesic(unweighted_ref.view(), weighted_ref.view()) > 0.3,
            "test design degenerate: weighted and unweighted means nearly coincide"
        );

        let (_t_uw, base_uw, _c) =
            dispatch_log_map(values.view(), "stiefel(k=1)", None, None).expect("unweighted chart");
        let (_t_w, base_w, _c) =
            dispatch_log_map(values.view(), "stiefel(k=1)", None, Some(weights.view()))
                .expect("weighted chart");

        // (a) Supplying weights must change the base point (before the fix the
        // weighted chart origin was byte-identical to the unweighted one).
        let moved = base_w
            .iter()
            .zip(base_uw.iter())
            .any(|(w, u)| (w - u).abs() > 1e-9);
        assert!(
            moved,
            "weighted base point is identical to the unweighted one: weights ignored"
        );

        // (b) The weighted base point must be closer to the WEIGHTED Fréchet
        // mean than to the unweighted one.
        let d_to_weighted = geodesic(base_w.view(), weighted_ref.view());
        let d_to_unweighted = geodesic(base_w.view(), unweighted_ref.view());
        assert!(
            d_to_weighted < d_to_unweighted,
            "weighted base point is nearer the unweighted mean ({d_to_unweighted}) \
             than the weighted mean ({d_to_weighted})"
        );
        // And it should essentially coincide with the weighted mean.
        assert!(
            d_to_weighted < 1e-6,
            "weighted base point is {d_to_weighted} from the weighted Fréchet mean"
        );
    }

    /// Deterministic xorshift64* + Box–Muller standard normals — a dependency-free
    /// reproducible source for the synthetic known-κ clouds. Seeded per call so
    /// the test is bit-stable across runs and platforms.
    struct DetNormal {
        state: u64,
        spare: Option<f64>,
    }
    impl DetNormal {
        fn new(seed: u64) -> Self {
            Self {
                state: seed | 1,
                spare: None,
            }
        }
        fn u01(&mut self) -> f64 {
            // xorshift64*; take the top 53 bits as a (0,1) double.
            let mut x = self.state;
            x ^= x >> 12;
            x ^= x << 25;
            x ^= x >> 27;
            self.state = x;
            let v = x.wrapping_mul(0x2545_F491_4F6C_DD1D);
            ((v >> 11) as f64 + 0.5) / (1u64 << 53) as f64
        }
        fn normal(&mut self) -> f64 {
            if let Some(z) = self.spare.take() {
                return z;
            }
            // Box–Muller; clamp u1 away from 0 so ln is finite.
            let u1 = self.u01().max(1e-12);
            let u2 = self.u01();
            let r = (-2.0 * u1.ln()).sqrt();
            let theta = 2.0 * std::f64::consts::PI * u2;
            self.spare = Some(r * theta.sin());
            r * theta.cos()
        }
    }

    /// Build a synthetic cloud at known curvature `k_star`: `n` points whose
    /// geodesic normal coordinates about `center` are i.i.d. isotropic Gaussian
    /// of scale `sigma`, exp-mapped onto `M_{k_star}`, then mean-centred in the
    /// ambient chart to mimic the real (mean-subtracted) response clouds.
    fn synth_cloud(dim: usize, k_star: f64, n: usize, sigma: f64, seed: u64) -> Array2<f64> {
        let manifold = ResponseManifold::ConstantCurvature { dim, kappa: k_star };
        let center = Array1::<f64>::zeros(dim);
        let mut rng = DetNormal::new(seed);
        let mut values = Array2::<f64>::zeros((n, dim));
        for i in 0..n {
            let t: Array1<f64> = (0..dim).map(|_| sigma * rng.normal()).collect();
            let y = manifold
                .exp_point(center.view(), t.view())
                .expect("exp tangent to response");
            values.row_mut(i).assign(&y);
        }
        // Mean-centre in the ambient chart (the real-data preprocessing).
        let mut mean = Array1::<f64>::zeros(dim);
        for row in values.outer_iter() {
            mean += &row;
        }
        mean.mapv_inplace(|v| v / n as f64);
        for mut row in values.outer_iter_mut() {
            row -= &mean;
        }
        values
    }

    /// The #1104 reparameterisation-invariant curvature estimator: on synthetic
    /// clouds generated at known κ⋆ the fitted κ̂ must be (a) INTERIOR to the
    /// chart bracket (never railed), (b) close to κ⋆ and MONOTONE in κ⋆, (c)
    /// produce a smooth (non-degenerate) χ²₁ flatness p-value that does not reject
    /// the flat truth, and (d) be correctly COVARIANT under a global rescaling of
    /// the cloud (κ has units 1/length², so `y ↦ α y ⇒ κ̂ ↦ κ̂/α²`).
    #[test]
    fn fit_response_curvature_is_reparameterization_invariant() {
        let dim = 3usize;
        // Unit-ish scale: σ=0.15 keeps every geodesic radius (≈ a few·σ) well
        // inside the κ-stereographic chart for the most hyperbolic κ⋆ = −1.5
        // (chart needs ‖y‖² < 1/1.5 ≈ 0.667).
        let sigma = 0.15;
        let n = 300usize;
        let k_stars = [-1.5_f64, -0.5, 0.0, 0.6, 1.2];
        let mut k_hats = Vec::new();
        for (idx, &k_star) in k_stars.iter().enumerate() {
            let values = synth_cloud(dim, k_star, n, sigma, 0xC0FFEE ^ (idx as u64 + 1));
            let (kmin, kmax, _rho) = response_kappa_bounds(values.view());
            let fit = fit_response_curvature(values.view(), dim, 0.95, 1e-12, 256)
                .expect("response curvature fit");
            k_hats.push(fit.kappa_hat);

            // (a) INTERIOR: κ̂ strictly inside the bracket, not railed to either end.
            let span = kmax - kmin;
            assert!(
                fit.kappa_hat > kmin + 0.02 * span && fit.kappa_hat < kmax - 0.02 * span,
                "κ⋆={k_star}: κ̂={} railed to bracket [{kmin}, {kmax}]",
                fit.kappa_hat
            );

            // (b-direct) recovery within a sane tolerance (finite-sample bias is
            // O(1/n); the estimator only needs the right region and sign).
            assert!(
                (fit.kappa_hat - k_star).abs() <= 0.6 + 0.3 * k_star.abs(),
                "κ⋆={k_star}: κ̂={} too far",
                fit.kappa_hat
            );

            // (c) the profile CI is a valid interval bracketing κ̂.
            assert!(
                fit.profile_ci.ci_lo <= fit.kappa_hat && fit.kappa_hat <= fit.profile_ci.ci_hi,
                "κ⋆={k_star}: CI [{}, {}] excludes κ̂={}",
                fit.profile_ci.ci_lo,
                fit.profile_ci.ci_hi,
                fit.kappa_hat
            );
            // The flatness LR statistic and p-value are valid; the p-value is a
            // genuine probability strictly between 0 and 1 (smooth, not 0/1).
            assert!(fit.flatness.lr_stat >= 0.0);
            assert!(
                fit.flatness.p_value > 0.0 && fit.flatness.p_value < 1.0,
                "κ⋆={k_star}: degenerate flatness p={}",
                fit.flatness.p_value
            );
            // The flat truth κ⋆ = 0 must NOT be rejected at 5% (lr < χ²_{1,.95}).
            if k_star == 0.0 {
                assert!(
                    fit.flatness.lr_stat < 3.84,
                    "flat truth wrongly rejected: lr={}",
                    fit.flatness.lr_stat
                );
            }

            // (d) RESCALING COVARIANCE: scale the SAME cloud by α and refit; κ̂
            // must transform as κ̂/α² (curvature has units 1/length²). We reuse the
            // identical points so the only change is the global scale.
            let alpha = 1.5_f64;
            let scaled = values.mapv(|v| alpha * v);
            let fit_scaled = fit_response_curvature(scaled.view(), dim, 0.95, 1e-12, 256)
                .expect("scaled response curvature fit");
            let expected = fit.kappa_hat / (alpha * alpha);
            // Tolerance scales with magnitude; the transform is exact in the
            // criterion (V(κ, αy) = V(α²κ, y)) up to the finite golden-section /
            // bracket discretisation.
            assert!(
                (fit_scaled.kappa_hat - expected).abs() <= 0.05 + 0.05 * expected.abs(),
                "κ⋆={k_star}: rescale covariance broken: κ̂(αy)={} vs κ̂(y)/α²={}",
                fit_scaled.kappa_hat,
                expected
            );
        }

        // (b-monotone) κ̂ is monotone increasing in κ⋆ across the whole sweep.
        for w in k_hats.windows(2) {
            assert!(w[1] > w[0] - 0.05, "κ̂ not monotone in κ⋆: {:?}", k_hats);
        }
    }

    /// d = 1 carries REDUCED curvature information: the transverse volume
    /// Jacobian is identically 1 (radial isometry), so κ is identified by the
    /// conformal-factor restoring force `−d·Σ ln λ_{y_i}` alone (#944 power
    /// analysis). The estimator must still run end-to-end, return an INTERIOR
    /// κ̂, and produce a valid CI — never divide/exponentiate the absent
    /// transverse direction.
    #[test]
    fn fit_response_curvature_d1_uses_conformal_term_only() {
        let sigma = 0.12;
        let n = 400usize;
        for &k_star in &[-1.0_f64, 0.0, 0.8] {
            let values = synth_cloud(1, k_star, n, sigma, 0xD1 ^ (k_star.to_bits()));
            let (kmin, kmax, _rho) = response_kappa_bounds(values.view());
            let fit = fit_response_curvature(values.view(), 1, 0.95, 1e-12, 256)
                .expect("d=1 curvature fit");
            let span = kmax - kmin;
            assert!(
                fit.kappa_hat > kmin + 0.01 * span && fit.kappa_hat < kmax - 0.01 * span,
                "d=1 κ⋆={k_star}: κ̂={} railed to [{kmin},{kmax}]",
                fit.kappa_hat
            );
            assert!(
                fit.profile_ci.ci_lo <= fit.kappa_hat && fit.kappa_hat <= fit.profile_ci.ci_hi,
                "d=1 κ⋆={k_star}: CI excludes κ̂"
            );
            assert!(fit.kappa_hat.is_finite() && fit.v_p_hat.is_finite());
        }
    }

    /// The criterion guard must reject κ probes AT or PAST the chart boundary
    /// gracefully (an `Err`, never a panic / NaN): on the hyperbolic edge
    /// `1 + κ‖y‖² ≤ 0` and on the spherical antipode. The `response_kappa_bounds`
    /// bracket stays strictly interior, but a stray CI/LR probe can land on the
    /// edge, so the criterion itself must be defensive.
    #[test]
    fn response_curvature_criterion_rejects_boundary_probes() {
        // A cloud with a known max radius R²; the hyperbolic edge is κ = −1/R².
        let values = array![[0.5_f64, 0.0], [-0.4, 0.3], [0.1, -0.5]];
        let r2_max = values
            .outer_iter()
            .map(|r| r.dot(&r))
            .fold(0.0_f64, f64::max);
        // Exactly on / past the hyperbolic edge: 1 + κ‖y‖² = 0 (or < 0).
        let kappa_edge = -1.0 / r2_max;
        assert!(
            response_curvature_criterion(values.view(), 2, kappa_edge).is_err(),
            "criterion must reject the hyperbolic chart edge κ=−1/R²"
        );
        assert!(
            response_curvature_criterion(values.view(), 2, 1.5 * kappa_edge).is_err(),
            "criterion must reject past the hyperbolic chart edge"
        );
        // Interior κ just inside the edge succeeds and is finite.
        let (v, _) = response_curvature_criterion(values.view(), 2, 0.9 * kappa_edge)
            .expect("interior κ valid");
        assert!(v.is_finite());
        // Non-finite κ is rejected up front.
        assert!(response_curvature_criterion(values.view(), 2, f64::NAN).is_err());
        assert!(response_curvature_criterion(values.view(), 2, f64::INFINITY).is_err());
    }

    // ── Projection residual (distance to candidate manifold) ───────────────

    #[test]
    fn projection_residual_is_zero_for_on_manifold_points() {
        // On-manifold rows are their own nearest point, so the residual is ~0
        // row-wise. No base point / Fréchet mean is involved — projection is
        // base-independent — so this no longer depends on the inputs forming an
        // admissible Karcher seed.
        let cases: Vec<(ResponseManifold, Array2<f64>)> = vec![
            (
                ResponseManifold::Spd { n: 2 }, // PD: eigenvalues {2,1} and {2,1}
                array![[2.0, 0.0, 0.0, 1.0], [1.5, 0.5, 0.5, 1.5]],
            ),
            (
                ResponseManifold::Grassmann { k: 1, n: 3 }, // unit columns
                array![[1.0, 0.0, 0.0], [0.6, 0.8, 0.0]],
            ),
            (
                ResponseManifold::Poincare {
                    dim: 2,
                    curvature: -1.0,
                }, // strictly inside the ball
                array![[0.1, 0.2], [-0.3, 0.1]],
            ),
        ];
        for (manifold, values) in cases {
            let (resid, rel) =
                response_projection_residual(manifold, values.view()).expect("projection residual");
            for row in 0..values.nrows() {
                assert!(
                    resid[row] < 1e-9,
                    "{manifold:?} on-manifold row {row} should have ~0 residual, got {}",
                    resid[row]
                );
                assert!(rel[row] < 1e-9 && rel[row] >= 0.0);
            }
        }
    }

    #[test]
    fn projection_residual_recovers_known_off_manifold_displacement() {
        // Closed-form checks against the exact nearest-point distance.

        // Gr(1,3) / sphere: nearest unit vector to x is x/‖x‖, so the distance
        // is |‖x‖ − 1|. [2,0,0] ⇒ 1; [0,3,0] ⇒ 2. Relative = dist/‖x‖.
        let g = ResponseManifold::Grassmann { k: 1, n: 3 };
        let gv = array![[2.0, 0.0, 0.0], [0.0, 3.0, 0.0]];
        let (gres, grel) = response_projection_residual(g, gv.view()).expect("grassmann");
        assert!((gres[0] - 1.0).abs() < 1e-12, "got {}", gres[0]);
        assert!((gres[1] - 2.0).abs() < 1e-12, "got {}", gres[1]);
        assert!((grel[0] - 0.5).abs() < 1e-12);
        assert!((grel[1] - 2.0 / 3.0).abs() < 1e-12);

        // SPD(2): nearest PSD matrix clamps negative eigenvalues to 0, so the
        // distance is the norm of the discarded negative part. [[1,0],[0,-1]]
        // has eigenvalue −1 discarded ⇒ distance 1; ‖x‖_F = √2.
        let s = ResponseManifold::Spd { n: 2 };
        let sv = array![[1.0, 0.0, 0.0, -1.0]];
        let (sres, srel) = response_projection_residual(s, sv.view()).expect("spd");
        assert!((sres[0] - 1.0).abs() < 1e-9, "got {}", sres[0]);
        assert!((srel[0] - 1.0 / 2.0_f64.sqrt()).abs() < 1e-9);

        // Poincaré ball (c = −1, true radius R = 1): the distance to the open
        // ball is max(0, ‖x‖ − R). [3,0] ⇒ exactly 2 (not 3 − (1 − BOUNDARY_EPS)
        // — the diagnostic uses the manifold radius, not the safety radius).
        let p = ResponseManifold::Poincare {
            dim: 2,
            curvature: -1.0,
        };
        let pv = array![[3.0, 0.0]];
        let (pres, _prel) = response_projection_residual(p, pv.view()).expect("poincare");
        assert!((pres[0] - 2.0).abs() < 1e-12, "got {}", pres[0]);

        // A different curvature (c = −4, R = 1/2): [2,0] ⇒ 2 − 0.5 = 1.5.
        let p4 = ResponseManifold::Poincare {
            dim: 2,
            curvature: -4.0,
        };
        let (p4res, _) =
            response_projection_residual(p4, array![[2.0, 0.0]].view()).expect("poincare c=-4");
        assert!((p4res[0] - 1.5).abs() < 1e-12, "got {}", p4res[0]);
    }

    #[test]
    fn projection_residual_validates_shapes_and_finiteness() {
        let manifold = ResponseManifold::Spd { n: 2 }; // ambient = 4
        // Wrong column count.
        let bad_cols = array![[1.0, 2.0, 3.0]];
        assert!(response_projection_residual(manifold, bad_cols.view()).is_err());
        // Non-finite value.
        let nan_vals = array![[f64::NAN, 0.0, 0.0, 1.0]];
        assert!(response_projection_residual(manifold, nan_vals.view()).is_err());
        let inf_vals = array![[f64::INFINITY, 0.0, 0.0, 1.0]];
        assert!(response_projection_residual(manifold, inf_vals.view()).is_err());
    }

    #[test]
    fn projection_residual_separates_on_and_off_manifold() {
        // The motivating case, now honestly answered: an on-manifold row sits
        // at zero distance from the candidate shape; a row pushed off it has a
        // clearly positive distance. This is the shape-plausibility signal that
        // gates which topology is worth fitting — not the post-fit membership
        // decision, which comes from the fitted surface's residual instead.
        let manifold = ResponseManifold::Grassmann { k: 1, n: 3 };
        let on = array![[0.6, 0.8, 0.0]]; // a genuine unit direction
        let off = array![[0.6, 0.8, 1.4]]; // same direction, pushed off-sphere

        let (resid_on, _) = response_projection_residual(manifold, on.view()).expect("on");
        let (resid_off, _) = response_projection_residual(manifold, off.view()).expect("off");

        assert!(
            resid_on[0] < 1e-9,
            "on-manifold should be ~0, got {}",
            resid_on[0]
        );
        assert!(
            resid_off[0] > 1e-2 && resid_off[0] > resid_on[0],
            "off-manifold distance ({}) must clearly exceed on-manifold ({})",
            resid_off[0],
            resid_on[0]
        );
    }

    #[test]
    fn projection_residual_supports_k_greater_than_one_frames() {
        // k > 1 frames use the closed form √Σ(σ_i − 1)². St(2,3), ambient = 6,
        // row-major n×k.
        let manifold = ResponseManifold::Stiefel { k: 2, n: 3 };

        // An orthonormal frame [e1 | e2] is its own nearest point ⇒ residual 0.
        let on = array![[1.0, 0.0, 0.0, 1.0, 0.0, 0.0]];
        let (resid_on, _) = response_projection_residual(manifold, on.view()).expect("on");
        assert!(
            resid_on[0] < 1e-9,
            "orthonormal frame should be ~0, got {}",
            resid_on[0]
        );

        // Scale the first column by 2: Y = [2·e1 | e2]. YᵀY = diag(4,1) ⇒
        // σ = (2,1), distance √((2−1)²+(1−1)²) = 1, relative = 1/‖Y‖_F = 1/√5.
        let off = array![[2.0, 0.0, 0.0, 1.0, 0.0, 0.0]];
        let (resid_off, rel_off) = response_projection_residual(manifold, off.view()).expect("off");
        assert!((resid_off[0] - 1.0).abs() < 1e-9, "got {}", resid_off[0]);
        assert!(
            (rel_off[0] - 1.0 / 5.0_f64.sqrt()).abs() < 1e-9,
            "got {}",
            rel_off[0]
        );

        // Grassmann(2,4) gives the identical score for the same frame data.
        let g = ResponseManifold::Grassmann { k: 2, n: 4 };
        let g_on = array![[1.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0]];
        let (g_resid, _) = response_projection_residual(g, g_on.view()).expect("grassmann");
        assert!(g_resid[0] < 1e-9, "got {}", g_resid[0]);
    }

    #[test]
    fn projection_residual_handles_nontrivial_eigenvectors() {
        // A frame whose Gram is NOT diagonal, so the singular values come from a
        // genuine eigendecomposition. Y = [[1,1],[0,1],[0,0]] (St(2,3)):
        // YᵀY = [[1,1],[1,2]], eigenvalues (3±√5)/2, σ = ((1+√5)/2, (√5−1)/2).
        // distance² = (σ₁−1)² + (σ₂−1)².
        let manifold = ResponseManifold::Stiefel { k: 2, n: 3 };
        let y = array![[1.0, 1.0, 0.0, 1.0, 0.0, 0.0]]; // row-major rows [1,1],[0,1],[0,0]
        let (resid, _) = response_projection_residual(manifold, y.view()).expect("frame");
        let s5 = 5.0_f64.sqrt();
        let sig1 = (1.0 + s5) / 2.0;
        let sig2 = (s5 - 1.0) / 2.0;
        let expect = ((sig1 - 1.0).powi(2) + (sig2 - 1.0).powi(2)).sqrt();
        assert!(
            (resid[0] - expect).abs() < 1e-9,
            "got {} want {}",
            resid[0],
            expect
        );
    }

    #[test]
    fn projection_residual_is_defined_for_rank_deficient_frames() {
        // A rank-deficient frame has a well-defined distance even though the
        // nearest orthonormal frame is not unique — distance to a compact set is
        // always defined, so this must NOT error. Two identical columns e1 give
        // YᵀY = [[1,1],[1,1]], σ = (√2, 0), distance √((√2−1)²+(0−1)²) = √(4−2√2).
        let manifold = ResponseManifold::Stiefel { k: 2, n: 3 };
        let degenerate = array![[1.0, 1.0, 0.0, 0.0, 0.0, 0.0]]; // both columns = e1
        let (resid, _) =
            response_projection_residual(manifold, degenerate.view()).expect("rank-deficient ok");
        let expect = (4.0 - 2.0 * 2.0_f64.sqrt()).sqrt(); // ≈ 1.0823922
        assert!(
            (resid[0] - expect).abs() < 1e-9,
            "got {} want {}",
            resid[0],
            expect
        );

        // Minimal case: zero vector on the sphere (Gr(1,3)). Every unit vector is
        // a nearest point and the distance is exactly 1 — also must not error.
        let sphere = ResponseManifold::Grassmann { k: 1, n: 3 };
        let (zres, _) =
            response_projection_residual(sphere, array![[0.0, 0.0, 0.0]].view()).expect("zero");
        assert!((zres[0] - 1.0).abs() < 1e-12, "got {}", zres[0]);
    }

    #[test]
    fn projection_residual_handles_tiny_full_rank_frame() {
        // A tiny but full-rank frame must NOT be rejected as rank-deficient: the
        // distance is scale-correct. Y = 1e-7·[e1 | e2] (St(2,3)) ⇒ σ = (1e-7,
        // 1e-7), distance √2·(1 − 1e-7) ≈ 1.41421342.
        let manifold = ResponseManifold::Stiefel { k: 2, n: 3 };
        let tiny = array![[1e-7, 0.0, 0.0, 1e-7, 0.0, 0.0]];
        let (resid, _) = response_projection_residual(manifold, tiny.view()).expect("tiny ok");
        let expect = 2.0_f64.sqrt() * (1.0 - 1e-7);
        assert!(
            (resid[0] - expect).abs() < 1e-9,
            "got {} want {}",
            resid[0],
            expect
        );
    }

    #[test]
    fn projection_residual_spd_nonsymmetric_and_singular() {
        // Non-symmetric input: A = [[1,1],[-1,1]] has sym(A) = I (no negative
        // part), but the distance to the PSD cone still counts the skew part:
        // ‖A − I‖_F = √2.
        let spd = ResponseManifold::Spd { n: 2 };
        let asym = array![[1.0, 1.0, -1.0, 1.0]]; // row-major [[1,1],[-1,1]]
        let (ares, _) = response_projection_residual(spd, asym.view()).expect("nonsym");
        assert!((ares[0] - 2.0_f64.sqrt()).abs() < 1e-9, "got {}", ares[0]);

        // A singular PSD matrix diag(1,0) is in the closed cone ⇒ distance 0
        // (even though it is not strictly positive definite).
        let singular = array![[1.0, 0.0, 0.0, 0.0]];
        let (sres, _) = response_projection_residual(spd, singular.view()).expect("singular psd");
        assert!(
            sres[0] < 1e-12,
            "singular PSD should be ~0, got {}",
            sres[0]
        );
    }

    #[test]
    fn projection_residual_poincare_interior_shell_is_zero() {
        // A point in the numerical safety shell R_safe < ‖x‖ < R is a genuine
        // interior point of the manifold ball, so it must score exactly 0 — the
        // diagnostic uses the true radius, not the projection safety radius.
        let p = ResponseManifold::Poincare {
            dim: 2,
            curvature: -1.0,
        };
        let shell = array![[0.999999, 0.0]]; // inside R = 1, outside R_safe ≈ 0.99999
        let (resid, _) = response_projection_residual(p, shell.view()).expect("shell");
        assert!(
            resid[0] < 1e-12,
            "interior point must be 0, got {}",
            resid[0]
        );
    }

    #[test]
    fn projection_residual_handles_constant_curvature_domain() {
        // ConstantCurvature is a fittable response geometry produced by the
        // resolver/parser, so it must return a closed-form distance, not error.
        // κ ≥ 0: chart is all of ℝ^d ⇒ every finite row scores 0.
        let pos = ResponseManifold::parse("constant_curvature(dim=3,kappa=1.0)", 3)
            .expect("parse constant_curvature");
        assert!(matches!(pos, ResponseManifold::ConstantCurvature { .. }));
        let (pres, _) =
            response_projection_residual(pos, array![[0.1, 9.0, -100.0]].view()).expect("kappa>=0");
        assert!(pres[0] < 1e-12, "κ≥0 finite row must be 0, got {}", pres[0]);

        // κ < 0: chart is the ball of radius 1/√(−κ). For κ = −1, R = 1, so a
        // point of norm 3 is at distance 2; an interior point is at 0.
        let neg = ResponseManifold::ConstantCurvature {
            dim: 2,
            kappa: -1.0,
        };
        let (nres, _) = response_projection_residual(neg, array![[3.0, 0.0], [0.2, 0.1]].view())
            .expect("kappa<0");
        assert!((nres[0] - 2.0).abs() < 1e-12, "got {}", nres[0]);
        assert!(nres[1] < 1e-12, "interior row must be 0, got {}", nres[1]);
    }

    #[test]
    fn projection_residual_accepts_empty_batch() {
        // A zero-row batch is valid and returns empty arrays for every geometry.
        let manifold = ResponseManifold::Spd { n: 2 }; // ambient = 4
        let empty = Array2::<f64>::zeros((0, 4));
        let (resid, rel) = response_projection_residual(manifold, empty.view()).expect("empty");
        assert_eq!(resid.len(), 0);
        assert_eq!(rel.len(), 0);
    }
}
