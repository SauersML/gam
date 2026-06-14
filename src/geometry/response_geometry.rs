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
//! the Poincaré [`exp_map`](crate::geometry::poincare::exp_map)/[`log_map`](crate::geometry::poincare::log_map));
//! the only new code is the batched row loop, the base-point dimension wiring,
//! and a generic Riemannian Karcher (Fréchet) mean shared by all four. There is
//! no separate per-manifold mean: the SPD safeguarded Karcher iteration is
//! generalised once, over the metric supplied by
//! [`RiemannianManifold::metric_tensor`], so adding a curved response geometry
//! is a single resolver arm.

use ndarray::{Array1, Array2, ArrayView1, ArrayView2};

use crate::geometry::constant_curvature::ConstantCurvature;
use crate::geometry::manifold::RiemannianManifold;
use crate::geometry::{GeometryResult, GrassmannManifold, SpdManifold, StiefelManifold};

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
                crate::geometry::poincare::log_map(base, value, *curvature)
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
                crate::geometry::poincare::exp_map(base, tangent, *curvature)
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
                let lam = crate::geometry::poincare::conformal_factor(base, *curvature)?;
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

/// String-driven response-geometry log map: parse the user `label` (with shape
/// inference from the response column count), pick the base point (intrinsic
/// Fréchet mean when `base` is `None`), map every row to its tangent, and report
/// the canonical resolved label. This is the curved-manifold analogue of the
/// sphere/simplex dispatch and the single entry the FFI calls for these
/// geometries.
pub fn dispatch_log_map(
    values: ArrayView2<'_, f64>,
    label: &str,
    base: Option<ArrayView1<'_, f64>>,
) -> Result<(Array2<f64>, Array1<f64>, String), String> {
    let manifold = ResponseManifold::parse(label, values.ncols())?;
    let base_point = match base {
        Some(b) => b.to_owned(),
        None => response_frechet_mean(manifold, values, None, 1.0e-12, 256)?,
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
/// version in [`crate::geometry::spd::spd_frechet_mean`] remains for the affine
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
    let w = crate::geometry::normalize_weights(m, weights)
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

    // Initialise at the projected ambient mean: a single log/exp round trip from
    // the first sample lands on the manifold for any geometry (the raw ambient
    // average is generally off-manifold for SPD/Grassmann/Stiefel/Poincaré).
    let mut p = manifold
        .exp_point(samples[0].view(), Array1::<f64>::zeros(ambient).view())
        .map_err(|e| format!("response geometry Fréchet mean init: {e}"))?;
    {
        // Average the log-tangents of all samples at the seed and step there:
        // one Riemannian averaging step gives an order-independent interior
        // start without assuming the ambient mean is on-manifold.
        let mut xi = Array1::<f64>::zeros(ambient);
        for (i, x) in samples.iter().enumerate() {
            let lg = manifold
                .log_point(p.view(), x.view())
                .map_err(|e| format!("response geometry Fréchet mean init log: {e}"))?;
            xi.scaled_add(w[i], &lg);
        }
        p = manifold
            .exp_point(p.view(), xi.view())
            .map_err(|e| format!("response geometry Fréchet mean init step: {e}"))?;
    }

    let mut f_cur = dispersion(p.view())?;
    let mut best_p = p.clone();
    let mut best_grad = f64::INFINITY;
    const STALL_REL: f64 = 5.0e-3;
    const STALL_PATIENCE: usize = 10;
    let mut stall = 0_usize;
    const ARMIJO_C1: f64 = 1.0e-4;
    const MAX_BACKTRACK_HALVINGS: usize = 60;
    const ARMIJO_ROUNDOFF_EPS_MULTIPLE: f64 = 8.0;

    for _ in 0..max_iter {
        let mut xi = Array1::<f64>::zeros(ambient);
        for (i, x) in samples.iter().enumerate() {
            let lg = manifold
                .log_point(p.view(), x.view())
                .map_err(|e| format!("response geometry Fréchet mean log map: {e}"))?;
            xi.scaled_add(w[i], &lg);
        }
        let grad_norm = manifold
            .sq_metric_norm(p.view(), xi.view())
            .map_err(|e| format!("response geometry Fréchet mean metric: {e}"))?
            .sqrt();
        if grad_norm <= tol {
            return Ok(p);
        }

        let improved = grad_norm < best_grad * (1.0 - STALL_REL);
        if grad_norm < best_grad {
            best_grad = grad_norm;
            best_p.assign(&p);
        }
        if improved {
            stall = 0;
        } else {
            stall += 1;
            if stall >= STALL_PATIENCE {
                return Ok(best_p);
            }
        }

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
            return Ok(best_p);
        }
    }
    Err("response geometry Fréchet mean did not reach stationarity within max_iter".into())
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
    pub kappa_hat: f64,
    /// The intrinsic Fréchet-mean base point at κ̂ (the tangent expansion point
    /// the scalar GAMs are fitted around).
    pub base: Array1<f64>,
    /// Profiled criterion value `V_p(κ̂)` (concentrated negative log-evidence).
    pub v_p_hat: f64,
    /// Profile-likelihood CI for κ and the geometry verdict from its sign.
    pub profile_ci: crate::geometry::curvature_estimand::KappaProfileCi,
    /// Interior-point χ²₁ likelihood-ratio test of flatness (κ = 0).
    pub flatness: crate::geometry::curvature_estimand::FlatnessTest,
}

/// Chart-validity bounds on κ for a constant-curvature response geometry built
/// from the supplied responses.
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
fn response_kappa_bounds(values: ArrayView2<'_, f64>) -> (f64, f64) {
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
        return (-1.0e6, 1.0e6);
    }
    // Keep a safety margin off the singular hyperbolic boundary.
    let kappa_min = if r2_max > 0.0 { -0.999 / r2_max } else { -1.0e6 };
    // Conjugate-radius cap: ρ_max = 2·max‖y_i − μ‖ is the κ=0 geodesic radius.
    let kappa_max = if s2_max > 0.0 {
        let rho_max = 2.0 * s2_max.sqrt();
        let edge = 0.9 * std::f64::consts::PI / rho_max;
        edge * edge
    } else {
        1.0e6
    };
    (kappa_min, kappa_max)
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
    for row in values.outer_iter() {
        // Geodesic radius s_i = d_κ(μ, y_i); also validates y_i is in-chart.
        let s = chart
            .distance(base.view(), row)
            .map_err(|e| format!("response curvature criterion distance: {e}"))?;
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
/// criterion and handed to [`profile_ci_walk`](crate::geometry::profile_ci_walk)
/// to size the initial Wald step; the CI itself is the exact χ²₁ profile crossing.
/// Flatness is the interior-point χ²₁ LR test
/// [`flatness_lr_test`](crate::geometry::flatness_lr_test). κ = 0 is an interior
/// point of the analytic `S^d ← ℝ^d → H^d` family, so no boundary correction is
/// applied. Returns the κ̂, its tangent base point, the profile CI, and the Wilks
/// flatness test for the fit summary.
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
    let (kappa_min, kappa_max) = response_kappa_bounds(values);

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

    let profile_ci = crate::geometry::curvature_estimand::profile_ci_walk(
        &mut v_p, kappa_hat, v_pp, kappa_min, kappa_max, level, ktol,
    )?;
    let flatness = crate::geometry::curvature_estimand::flatness_lr_test(&mut v_p, kappa_hat)?;

    Ok(ResponseCurvatureFit {
        dim,
        kappa_hat,
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
                dispatch_log_map(values.view(), label, None).expect("dispatch log");
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
        let manifold = ResponseManifold::ConstantCurvature {
            dim,
            kappa: k_star,
        };
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
            let (kmin, kmax) = response_kappa_bounds(values.view());
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
                (fit_scaled.kappa_hat - expected).abs()
                    <= 0.05 + 0.05 * expected.abs(),
                "κ⋆={k_star}: rescale covariance broken: κ̂(αy)={} vs κ̂(y)/α²={}",
                fit_scaled.kappa_hat,
                expected
            );
        }

        // (b-monotone) κ̂ is monotone increasing in κ⋆ across the whole sweep.
        for w in k_hats.windows(2) {
            assert!(
                w[1] > w[0] - 0.05,
                "κ̂ not monotone in κ⋆: {:?}",
                k_hats
            );
        }
    }
}
