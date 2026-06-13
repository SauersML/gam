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
                let kappa = get_f64("kappa")?.or_else(|| get_f64("curvature").ok().flatten()).unwrap_or(0.0);
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
            _ => self
                .riemannian()
                .expect("matrix response manifold")
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
            _ => self
                .riemannian()
                .expect("matrix response manifold")
                .exp_map(base, tangent),
        }
    }

    /// Squared metric norm `‖v‖²_base` of a tangent at `base`. Used by the
    /// Karcher iteration's stationarity test. Poincaré uses the conformal
    /// factor squared; the matrix manifolds use the trait metric tensor.
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
            _ => {
                let g = self
                    .riemannian()
                    .expect("matrix response manifold")
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
}
