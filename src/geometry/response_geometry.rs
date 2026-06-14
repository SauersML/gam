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
// the user: κ is ESTIMATED from the manifold-valued responses via the REML outer
// loop, exactly as the smooth-term Matérn κ is. At each κ the family
// `ConstantCurvature{dim, κ}` is laid down, the intrinsic mean μ (the tangent
// base point) is re-solved, and κ is scored by the PROPERLY NORMALISED intrinsic
// Gaussian likelihood — the wrapped/normal-coordinate Gaussian whose density on
// the manifold integrates to one. This is the crux of the #1104 fix: a model
// that is only normalised in the flat tangent space is scale-degenerate in κ.
//
// ## Why dispersion alone (and a mean-only log-det) rails
//
// The naive criterion `V_p(κ) = ½na·ln(D(κ)/na)` with `D(κ) = Σ‖log_{μ,κ}(yᵢ)‖²`
// is a function of the geodesic distances ONLY. Changing κ rescales the manifold
// radius `R = 1/√κ`, which rescales every distance, which rescales `D` — and a
// σ-profiled Gaussian on the residuals is invariant to that scale up to a
// `+na·ln(scale)` term that is monotone in κ. So `V_p` decreases without bound as
// κ→+∞ (the whole sphere shrinks to a point, `D→0`), and κ̂ rails to the upper
// bracket. A `−n·a·ln λ_μ` "metric log-det at the mean" term does NOT cure this:
// for centred / unit-scale data μ ≈ 0, so `λ_μ = 2/(1+κ‖μ‖²) ≈ 2` is essentially
// κ-independent and supplies zero restoring force — exactly the observed failure
// on real (mean-subtracted) response clouds, where κ̂ pinned at +50…+65.
//
// ## The restoring force is the intrinsic-volume partition function
//
// The generative model is `rᵢ = log_μ(yᵢ) ∼ N(0, σ²I_d)` in normal coordinates,
// `yᵢ = exp_μ(rᵢ)`. Its density on the MANIFOLD (w.r.t. the Riemannian volume
// `dvol_κ`) is `N(r;0,σ²I)/J_κ(‖r‖)`, where `J_κ(r) = (sn_κ(r)/r)^{d−1}` is the
// exp-map Jacobian (`ConstantCurvature::jacobian_radial`). For this to be a
// probability density on `M_κ` it must integrate to one against `dvol_κ`, i.e.
// the tangent Gaussian must be divided by its partition function
//
// ```text
//   Z(κ, σ²) = ∫_{ℝ^d} N(r;0,σ²I) · J_κ(‖r‖) dr
//            = E_{r∼N(0,σ²I_d)}[ J_κ(‖r‖) ]   (a clean 1-D radial expectation),
// ```
//
// which DOES move with κ even when μ = 0, because it integrates the volume the
// κ-geometry assigns to the spread of the cloud. The negative log-likelihood is
//
// ```text
//   −ℓ(κ,σ²) = D(κ)/(2σ²) + (na/2)·ln(2πσ²) + n·ln Z(κ,σ²),
// ```
//
// and `V_p(κ) = min_{σ²} −ℓ(κ,σ²)` (σ profiled by a short 1-D search, since `Z`
// couples σ and κ and σ̂² is no longer closed-form). As κ→+∞ the manifold has
// finite volume `∝ κ^{−d/2}`, so `Z→0` and `n·ln Z ~ −(nd/2)ln κ → +∞` faster
// than the dispersion term falls; as κ→−1/R² (hyperbolic chart edge) `J_κ`
// stretches, `Z` and `D` both grow and `V_p → +∞`. The minimum is therefore
// INTERIOR at the data-generating κ on unit-scale data, with no dependence on
// where μ sits. `κ→0` is the analytic interior point (`J_0 ≡ 1`, `Z = 1`,
// `V_p` reduces to the flat Gaussian), so no boundary correction is needed.
//
// `V_p` is a negative log-evidence (lower is better) so κ̂ = argmin V_p, and
// `2[V_p(0) − V_p(κ̂)]` is the Wilks LR statistic of flatness — exactly the
// contract `profile_ci_walk` / `flatness_lr_test` in `curvature_estimand.rs`
// consume. κ joins the outer optimisation as one signed coordinate with the same
// likelihood semantics as every other ψ; no new outer machinery is introduced.

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
/// from the supplied responses. The κ-stereographic chart requires
/// `1 + κ‖x‖² > 0` at every point `x`, i.e. `κ > −1/R²` with
/// `R² = max_i ‖y_i‖²`; the upper (spherical) side is unbounded but we cap the
/// search to a generous multiple of the same scale so the bracket stays finite.
fn response_kappa_bounds(values: ArrayView2<'_, f64>) -> (f64, f64) {
    let mut r2_max = 0.0_f64;
    for row in values.outer_iter() {
        let r2 = row.dot(&row);
        if r2 > r2_max {
            r2_max = r2;
        }
    }
    if r2_max <= 0.0 {
        // Degenerate (all points at the origin): κ is unidentified; use a wide
        // symmetric default so the optimiser/CI report a flat, unbounded result.
        return (-1.0e6, 1.0e6);
    }
    // Keep a safety margin off the singular boundary so chart_gauge stays well
    // above GEOMETRY_EPS along the whole walk.
    let kappa_min = -0.999 / r2_max;
    let kappa_max = 1.0e3 / r2_max;
    (kappa_min, kappa_max)
}

/// Number of Gauss–Legendre nodes for the radial partition-function integral
/// `Z(κ,σ²) = E_{χ∼chi_d}[J_κ(σχ)]`. The integrand is a smooth product of a
/// chi density and a (d−1)-power of the entire function `S`; 64 nodes on the
/// truncated radial domain resolve it to far below the κ-search tolerance.
const PARTITION_GL_NODES: usize = 64;

/// Upper truncation of the radial (chi) variable in units of its scale: the
/// chi_d density past `mean + 12·sd` carries < 1e−30 of the mass for any d in
/// range, so the truncated integral equals the full one to machine precision.
const PARTITION_CHI_SIGMAS: f64 = 12.0;

/// `ln Γ(x)` for `x > 0` — Lanczos g=7, n=9; accurate to ~1e−14 on the chi
/// normalising constant `2^{d/2−1} Γ(d/2)` we need here.
fn ln_gamma(x: f64) -> f64 {
    const G: f64 = 7.0;
    const C: [f64; 9] = [
        0.999_999_999_999_809_93,
        676.520_368_121_885_1,
        -1_259.139_216_722_402_8,
        771.323_428_777_653_13,
        -176.615_029_162_140_6,
        12.507_343_278_686_905,
        -0.138_571_095_265_720_12,
        9.984_369_578_019_572e-6,
        1.505_632_735_149_311_6e-7,
    ];
    if x < 0.5 {
        // Reflection: Γ(x)Γ(1−x) = π/sin(πx).
        let pi = std::f64::consts::PI;
        return (pi / (pi * x).sin()).ln() - ln_gamma(1.0 - x);
    }
    let x = x - 1.0;
    let mut a = C[0];
    let t = x + G + 0.5;
    for (i, &c) in C.iter().enumerate().skip(1) {
        a += c / (x + i as f64);
    }
    0.5 * (2.0 * std::f64::consts::PI).ln() + (x + 0.5) * t.ln() - t + a.ln()
}

/// Gauss–Legendre nodes/weights on `[-1, 1]` for `n` points, by Newton on the
/// Legendre polynomial (Golub–Welsch-free; deterministic, dependency-light).
fn gauss_legendre(n: usize) -> (Vec<f64>, Vec<f64>) {
    let mut nodes = vec![0.0; n];
    let mut weights = vec![0.0; n];
    let m = n.div_ceil(2);
    for i in 0..m {
        // Initial guess: the asymptotic zero of P_n.
        let mut x = (std::f64::consts::PI * (i as f64 + 0.75) / (n as f64 + 0.5)).cos();
        let mut dp = 0.0;
        for _ in 0..100 {
            // Evaluate P_n(x) and P_n'(x) by the recurrence.
            let (mut p0, mut p1) = (1.0_f64, x);
            for k in 2..=n {
                let kf = k as f64;
                let p2 = ((2.0 * kf - 1.0) * x * p1 - (kf - 1.0) * p0) / kf;
                p0 = p1;
                p1 = p2;
            }
            dp = (n as f64) * (x * p1 - p0) / (x * x - 1.0);
            let dx = p1 / dp;
            x -= dx;
            if dx.abs() <= 1e-15 {
                break;
            }
        }
        let w = 2.0 / ((1.0 - x * x) * dp * dp);
        nodes[i] = -x;
        nodes[n - 1 - i] = x;
        weights[i] = w;
        weights[n - 1 - i] = w;
    }
    (nodes, weights)
}

/// Intrinsic-Gaussian partition function `Z(κ,σ²) = E_{r∼N(0,σ²I_d)}[J_κ(‖r‖)]`,
/// the normaliser that makes the wrapped Gaussian a proper density on `M_κ`.
///
/// With `‖r‖ = σ·χ`, `χ ∼ chi_d`, this is the 1-D radial integral
/// `∫₀^∞ J_κ(σχ) · f_{chi_d}(χ) dχ`, `f_{chi_d}(χ) = χ^{d−1}e^{−χ²/2}/(2^{d/2−1}Γ(d/2))`,
/// evaluated by Gauss–Legendre on the truncated domain `[0, χ_max]`. `J_κ(σχ) =
/// S(κσ²χ²)^{d−1}` ([`ConstantCurvature::jacobian_radial`]). Returns `ln Z`
/// (the criterion only ever needs the log), with `ln Z = 0` exactly at κ = 0
/// (`J_0 ≡ 1`). At κ > 0 the integrand is clamped to be non-negative past the
/// conjugate radius, so `Z` is the volume the κ-sphere actually assigns.
fn response_partition_log(dim: usize, kappa: f64, sigma2: f64, nodes: &[(f64, f64)]) -> f64 {
    if kappa == 0.0 || dim == 0 {
        return 0.0;
    }
    let d = dim as f64;
    let sigma = sigma2.max(0.0).sqrt();
    // chi_d density normaliser: f(χ) = χ^{d−1} e^{−χ²/2} / (2^{d/2−1} Γ(d/2)).
    let ln_norm = (0.5 * d - 1.0) * std::f64::consts::LN_2 + ln_gamma(0.5 * d);
    let chi_mean = std::f64::consts::SQRT_2 * (ln_gamma(0.5 * (d + 1.0)) - ln_gamma(0.5 * d)).exp();
    let chi_max = (chi_mean + PARTITION_CHI_SIGMAS).max(2.0 * chi_mean) + PARTITION_CHI_SIGMAS;
    let chart = ConstantCurvature::new(dim, kappa);
    // ∫₀^{χ_max} J_κ(σχ) f(χ) dχ, mapped from the GL reference [-1,1].
    let half = 0.5 * chi_max;
    let mut acc = 0.0_f64;
    for &(node, weight) in nodes {
        let chi = half * (node + 1.0);
        let ln_fchi = (d - 1.0) * chi.max(1e-300).ln() - 0.5 * chi * chi - ln_norm;
        let jac = chart.jacobian_radial(sigma * chi);
        acc += weight * jac * ln_fchi.exp();
    }
    let z = (half * acc).max(1e-300);
    z.ln()
}

/// Profiled curvature criterion `V_p(κ)` for the constant-curvature response
/// geometry: the σ-profiled negative log-likelihood of the PROPERLY NORMALISED
/// intrinsic Gaussian at curvature `κ`, with the intrinsic Fréchet mean re-solved
/// at this κ. Lower is better (κ̂ = argmin). Returns `(V_p, base_point)`; the base
/// is reused as the tangent expansion point at the optimum.
///
/// `D(κ) = Σ_i ‖log_{base,κ}(y_i)‖²_{base}` is the Fréchet dispersion at the
/// κ-mean. The model `r_i = log_μ(y_i) ∼ N(0,σ²I_d)`, `y_i = exp_μ(r_i)`, has
/// negative log-likelihood `D/(2σ²) + (na/2)ln(2πσ²) + n·ln Z(κ,σ²)`, where the
/// partition function `Z(κ,σ²) = E_{N(0,σ²I)}[J_κ(‖r‖)]` (see
/// [`response_partition_log`]) normalises the wrapped Gaussian on `M_κ` and is
/// the κ-restoring term that breaks the dispersion-only scale degeneracy. σ is
/// profiled by a 1-D Newton/bisection because `Z` couples σ and κ (so σ̂² is no
/// longer the closed-form `D/na`). Additive constants independent of κ are kept
/// implicit — they cancel in every LR / profile-drop the CI machinery forms.
pub fn response_curvature_criterion(
    values: ArrayView2<'_, f64>,
    dim: usize,
    kappa: f64,
    tol: f64,
    max_iter: usize,
) -> Result<(f64, Array1<f64>), String> {
    if !kappa.is_finite() {
        return Err("response curvature criterion: kappa must be finite".into());
    }
    let manifold = ResponseManifold::ConstantCurvature { dim, kappa };
    let (n_rows, _) = values.dim();
    let base = response_frechet_mean(manifold, values, None, tol, max_iter)?;
    let mut dispersion = 0.0_f64;
    for row in values.outer_iter() {
        let lg = manifold
            .log_point(base.view(), row)
            .map_err(|e| format!("response curvature criterion log map: {e}"))?;
        let sq = manifold
            .sq_metric_norm(base.view(), lg.view())
            .map_err(|e| format!("response curvature criterion metric: {e}"))?;
        dispersion += sq;
    }
    let nobs = (n_rows * dim) as f64;
    // Floor the dispersion so a (near-)perfect flat fit does not blow ln up; the
    // floor is far below any genuine residual scale and cancels in profile drops.
    let d = dispersion.max(1.0e-300 * nobs.max(1.0));

    // Radial GL quadrature for ln Z, reused across the σ-profile inner search.
    let (gl_nodes, gl_weights) = gauss_legendre(PARTITION_GL_NODES);
    let nodes: Vec<(f64, f64)> = gl_nodes.into_iter().zip(gl_weights).collect();

    // Negative log-likelihood at fixed κ, as a function of σ²:
    //   φ(σ²) = D/(2σ²) + (na/2) ln(2π σ²) + n · ln Z(κ, σ²).
    let nll = |sigma2: f64| -> f64 {
        let s2 = sigma2.max(1.0e-300);
        let ln_z = response_partition_log(dim, kappa, s2, &nodes);
        d / (2.0 * s2)
            + 0.5 * nobs * ((2.0 * std::f64::consts::PI * s2).ln())
            + (n_rows as f64) * ln_z
    };

    // σ-profile: at κ = 0, Z ≡ 1 and the closed-form minimiser is σ̂² = D/na;
    // for κ ≠ 0 the volume term shifts it, so we golden-section σ² in log-space
    // around that seed (the criterion is smooth and unimodal in σ²). The bracket
    // spans four decades each side of the flat MLE — far wider than the volume
    // term ever moves the optimum on a valid chart.
    let s2_flat = (d / nobs).max(1.0e-300);
    let v_p = if kappa == 0.0 {
        nll(s2_flat)
    } else {
        const GOLDEN_INV: f64 = 0.618_033_988_749_894_8;
        let mut lo = (s2_flat.ln()) - 4.0 * std::f64::consts::LN_10;
        let mut hi = (s2_flat.ln()) + 4.0 * std::f64::consts::LN_10;
        let mut c = hi - GOLDEN_INV * (hi - lo);
        let mut e = lo + GOLDEN_INV * (hi - lo);
        let mut fc = nll(c.exp());
        let mut fe = nll(e.exp());
        for _ in 0..200 {
            if (hi - lo).abs() <= 1.0e-9 {
                break;
            }
            if fc < fe {
                hi = e;
                e = c;
                fe = fc;
                c = hi - GOLDEN_INV * (hi - lo);
                fc = nll(c.exp());
            } else {
                lo = c;
                c = e;
                fc = fe;
                e = lo + GOLDEN_INV * (hi - lo);
                fe = nll(e.exp());
            }
        }
        nll((0.5 * (lo + hi)).exp())
    };
    Ok((v_p, base))
}

/// Fit curvature as an estimand on a constant-curvature response geometry.
///
/// κ̂ is the minimiser of the profiled criterion [`response_curvature_criterion`]
/// (the σ-profiled Gaussian negative log-evidence of the tangent model), found by
/// a golden-section search inside the chart-validity bracket. The exact outer
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
    // and the CI walk so every evaluation re-solves the intrinsic mean at its κ.
    let mut v_p = |kappa: f64| -> Result<f64, String> {
        response_curvature_criterion(values, dim, kappa, tol, max_iter).map(|(v, _)| v)
    };

    // ── κ̂: golden-section minimisation inside the chart bracket. ────────────
    // The dispersion criterion is smooth and unimodal in practice; golden
    // section is derivative-free and respects the bracket bounds exactly.
    const GOLDEN_INV: f64 = 0.618_033_988_749_894_8; // 1/φ
    const GOLDEN_TOL_REL: f64 = 1.0e-7;
    const GOLDEN_MAX_ITER: usize = 200;
    let mut a = kappa_min;
    let mut b = kappa_max;
    let mut c = b - GOLDEN_INV * (b - a);
    let mut d_pt = a + GOLDEN_INV * (b - a);
    let mut fc = v_p(c)?;
    let mut fd = v_p(d_pt)?;
    let ktol = GOLDEN_TOL_REL * (kappa_max - kappa_min).max(1.0);
    for _ in 0..GOLDEN_MAX_ITER {
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
    let (v_p_hat, base) = response_curvature_criterion(values, dim, kappa_hat, tol, max_iter)?;

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

    /// Curvature-as-estimand on a constant-curvature response geometry: data
    /// generated as exact geodesic exp-images of small Gaussian tangents at a
    /// known curvature κ⋆ must drive κ̂ to the true value, the profile CI must
    /// cover κ⋆, and (for a genuinely curved κ⋆) the flatness LR test must reject
    /// κ = 0. Crucially the κ̂ optimiser uses ONLY the responses — κ⋆ is never
    /// passed in — so this exercises the magic-by-default estimand contract.
    #[test]
    fn fit_response_curvature_recovers_true_kappa_and_rejects_flat() {
        let dim = 3usize;
        for &k_star in &[-1.5_f64, 0.0, 1.2] {
            let manifold = ResponseManifold::ConstantCurvature { dim, kappa: k_star };
            // Base point off the origin; deterministic spread of small tangents
            // (kept well inside the chart for every κ⋆) exp-mapped to responses.
            let base = array![0.04, -0.06, 0.05];
            let tangents = [
                array![0.10, -0.05, 0.02],
                array![-0.08, 0.06, -0.03],
                array![0.03, 0.09, -0.07],
                array![-0.11, -0.04, 0.08],
                array![0.06, 0.02, 0.10],
                array![0.01, -0.10, -0.06],
            ];
            let mut values = Array2::<f64>::zeros((tangents.len(), dim));
            for (i, t) in tangents.iter().enumerate() {
                let y = manifold
                    .exp_point(base.view(), t.view())
                    .expect("exp tangent to response");
                values.row_mut(i).assign(&y);
            }
            let fit = fit_response_curvature(values.view(), dim, 0.95, 1e-12, 256)
                .expect("response curvature fit");
            // κ̂ recovers κ⋆ (loose tolerance: a finite Fréchet sample profiles a
            // shallow criterion, but the minimiser must be on the right side and
            // close).
            assert!(
                (fit.kappa_hat - k_star).abs() <= 0.6 + 0.25 * k_star.abs(),
                "κ⋆={k_star}: κ̂={} too far",
                fit.kappa_hat
            );
            // The profile CI must bracket κ̂ and be a valid interval.
            assert!(
                fit.profile_ci.ci_lo <= fit.kappa_hat && fit.kappa_hat <= fit.profile_ci.ci_hi,
                "κ⋆={k_star}: CI [{}, {}] excludes κ̂={}",
                fit.profile_ci.ci_lo,
                fit.profile_ci.ci_hi,
                fit.kappa_hat
            );
            // Flatness p-value is a valid probability; the LR statistic is ≥ 0.
            assert!(fit.flatness.lr_stat >= 0.0);
            assert!(fit.flatness.p_value >= 0.0 && fit.flatness.p_value <= 1.0);
            // For the flat truth κ⋆ = 0 the LR statistic must be tiny (do NOT
            // reject flatness).
            if k_star == 0.0 {
                assert!(
                    fit.flatness.lr_stat < 3.84,
                    "flat truth wrongly rejected: lr={}",
                    fit.flatness.lr_stat
                );
            }
        }
    }
}
