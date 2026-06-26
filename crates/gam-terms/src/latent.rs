//! Pure latent-coordinate types used by basis-side input-location derivatives.

use crate::basis::{BasisError, RadialScalarKind};
use ndarray::{Array1, Array2, Array3, ArrayView1, ArrayView2, ArrayView3};

/// Natural manifold for per-row latent-coordinate updates.
///
/// `Euclidean` preserves the original additive update. `Circle` is a scalar
/// angular coordinate wrapped modulo `2π`. `Sphere { dim }` is the embedded
/// unit sphere in `R^dim`, with retraction `(t + ξ) / ||t + ξ||`. `Product`
/// composes these blockwise; inside a product, `Euclidean` denotes one
/// unconstrained scalar axis.
#[derive(Debug, Clone, PartialEq, Default)]
pub enum LatentManifold {
    /// Unconstrained `R^d` — the current default.
    #[default]
    Euclidean,
    /// Scalar periodic coordinate on `S^1` with caller-supplied period.
    ///
    /// Wraps modulo `period`; pass `period = 2π` for radian conventions and
    /// `period = 1.0` for basis evaluators that interpret the latent as a
    /// fraction of one period. The metric weight uses `1/period²` so the
    /// trust-region radius respects the chosen unit.
    Circle { period: f64 },
    /// Embedded unit sphere `S^(dim-1)`.
    Sphere { dim: usize },
    /// Closed interval in `R`; the retraction clamps to the boundary.
    Interval { lo: f64, hi: f64 },
    /// Product manifold, split block-by-block in row-major ambient storage.
    Product(Vec<LatentManifold>),
    /// Product manifold with explicit per-axis trust-region metric weights.
    ///
    /// Without per-axis weighting, a Product of Circle + Interval treats
    /// 1 radian as commensurate with the entire bounded range. With weights
    /// = 1/scale², the trust-region radius respects each axis's natural unit.
    ProductWithMetric {
        manifolds: Vec<LatentManifold>,
        weights: Vec<f64>,
    },
}

impl LatentManifold {
    pub fn is_euclidean(&self) -> bool {
        matches!(self, Self::Euclidean)
    }

    pub fn ambient_dim(&self, fallback_dim: usize) -> usize {
        match self {
            Self::Euclidean => fallback_dim,
            Self::Circle { .. } | Self::Interval { .. } => 1,
            Self::Sphere { dim } => *dim,
            Self::Product(parts)
            | Self::ProductWithMetric {
                manifolds: parts, ..
            } => parts.iter().map(|part| part.ambient_dim(1)).sum(),
        }
    }

    /// Project an arbitrary ambient point back to the manifold.
    pub fn project_point(&self, t: ArrayView1<'_, f64>) -> Array1<f64> {
        match self {
            Self::Euclidean => t.to_owned(),
            Self::Circle { period } => {
                let mut out = Array1::<f64>::zeros(1);
                out[0] = wrap_to_period(t[0], *period);
                out
            }
            Self::Sphere { dim } => {
                assert_eq!(t.len(), *dim);
                normalize_or_axis(t, *dim)
            }
            Self::Interval { lo, hi } => {
                // Order the bounds defensively: `f64::clamp` panics if min > max,
                // so a reversed `Interval { lo, hi }` would otherwise crash deep
                // in projection rather than clamp into the intended range.
                let (lo, hi) = if lo <= hi { (*lo, *hi) } else { (*hi, *lo) };
                let mut out = Array1::<f64>::zeros(1);
                out[0] = t[0].clamp(lo, hi);
                out
            }
            Self::Product(parts)
            | Self::ProductWithMetric {
                manifolds: parts, ..
            } => {
                let mut out = Array1::<f64>::zeros(t.len());
                let mut offset = 0_usize;
                for part in parts {
                    let dim = part.ambient_dim(1);
                    let projected = part.project_point(t.slice(ndarray::s![offset..offset + dim]));
                    for a in 0..dim {
                        out[offset + a] = projected[a];
                    }
                    offset += dim;
                }
                assert_eq!(offset, t.len());
                out
            }
        }
    }

    /// Retraction `R_t(ξ)`, using closed-form analytic maps for every variant.
    pub fn retract(&self, t: ArrayView1<'_, f64>, xi: ArrayView1<'_, f64>) -> Array1<f64> {
        assert_eq!(t.len(), xi.len());
        match self {
            Self::Euclidean => {
                let mut out = t.to_owned();
                for a in 0..out.len() {
                    out[a] += xi[a];
                }
                out
            }
            Self::Circle { period } => {
                let mut out = Array1::<f64>::zeros(1);
                out[0] = wrap_to_period(t[0] + xi[0], *period);
                out
            }
            Self::Sphere { dim } => {
                assert_eq!(t.len(), *dim);
                let mut y = Array1::<f64>::zeros(*dim);
                for a in 0..*dim {
                    y[a] = t[a] + xi[a];
                }
                normalize_or_axis(y.view(), *dim)
            }
            Self::Interval { lo, hi } => {
                // Order the bounds defensively: `f64::clamp` panics if min > max,
                // so a reversed `Interval { lo, hi }` would otherwise crash the
                // retraction instead of clamping into the intended range.
                let (lo, hi) = if lo <= hi { (*lo, *hi) } else { (*hi, *lo) };
                let mut out = Array1::<f64>::zeros(1);
                out[0] = (t[0] + xi[0]).clamp(lo, hi);
                out
            }
            Self::Product(parts)
            | Self::ProductWithMetric {
                manifolds: parts, ..
            } => {
                let mut out = Array1::<f64>::zeros(t.len());
                let mut offset = 0_usize;
                for part in parts {
                    let dim = part.ambient_dim(1);
                    let next = part.retract(
                        t.slice(ndarray::s![offset..offset + dim]),
                        xi.slice(ndarray::s![offset..offset + dim]),
                    );
                    for a in 0..dim {
                        out[offset + a] = next[a];
                    }
                    offset += dim;
                }
                assert_eq!(offset, t.len());
                out
            }
        }
    }

    /// Orthogonal projection of an ambient vector onto `T_t M`.
    pub fn project_to_tangent(
        &self,
        t: ArrayView1<'_, f64>,
        v: ArrayView1<'_, f64>,
    ) -> Array1<f64> {
        assert_eq!(t.len(), v.len());
        match self {
            Self::Euclidean | Self::Circle { .. } => v.to_owned(),
            Self::Sphere { dim } => {
                assert_eq!(t.len(), *dim);
                let tv = dot_views(t, v);
                let mut out = v.to_owned();
                for a in 0..*dim {
                    out[a] -= tv * t[a];
                }
                out
            }
            Self::Interval { lo, hi } => {
                let mut out = Array1::<f64>::zeros(1);
                let at_lo = t[0] <= *lo && v[0] < 0.0;
                let at_hi = t[0] >= *hi && v[0] > 0.0;
                out[0] = if at_lo || at_hi { 0.0 } else { v[0] };
                out
            }
            Self::Product(parts)
            | Self::ProductWithMetric {
                manifolds: parts, ..
            } => {
                let mut out = Array1::<f64>::zeros(v.len());
                let mut offset = 0_usize;
                for part in parts {
                    let dim = part.ambient_dim(1);
                    let projected = part.project_to_tangent(
                        t.slice(ndarray::s![offset..offset + dim]),
                        v.slice(ndarray::s![offset..offset + dim]),
                    );
                    for a in 0..dim {
                        out[offset + a] = projected[a];
                    }
                    offset += dim;
                }
                assert_eq!(offset, v.len());
                out
            }
        }
    }
}

/// Carrier for the `∂Φ/∂t` chain-rule input, dispatched on basis kind by
/// [`LatentCoordValues::design_gradient_wrt_t_dispatch`].
///
/// * [`InputLocationDerivative::Radial`] is the *radial-kernel* path: the
///   caller supplies the radial kernel family together with the center
///   coordinates, and the chain rule
///   `∂Φ/∂t = q(r) · (t − c)` is applied internally. This covers every
///   isotropic radial basis — Duchon (any nullspace order), Matérn (every
///   supported half-integer ν), and anything else whose pointwise
///   gradient is radial. Helpers:
///   [`crate::basis::duchon_radial_first_derivative_nd`],
///   [`crate::basis::matern_radial_first_derivative_nd`].
/// * [`InputLocationDerivative::Jet`] is the *pre-computed jet* path: the
///   caller has already assembled a closed-form `(N, K, d)` tensor for a
///   basis whose chain rule is not a simple radial scalar times a unit
///   vector. Sphere kernels carry the tangent-direction times `K'(cos γ)`;
///   periodic-cyclic B-splines carry the closed-form cardinal derivative;
///   tensor-product B-splines carry the product-rule mix. Helpers:
///   [`crate::basis::sphere_first_derivative_nd`],
///   [`crate::basis::periodic_bspline_first_derivative_nd`],
///   [`crate::basis::bspline_tensor_first_derivative`].
///
/// The dispatch is an enum rather than a trait because each path's
/// arguments differ structurally (radial bases reuse scalar radial kernels shared with
/// the kernel-shape chain machinery; jet bases ship the full tensor). All chain rules
/// are analytic and closed-form; no autodiff, no finite differences.
pub enum InputLocationDerivative<'a> {
    /// Radial-kernel chain rule. The chain rule `(t − c)/r` is reconstructed
    /// internally from the finite `q = φ'(r)/r` scalar and the center coordinates.
    Radial {
        centers: ArrayView2<'a, f64>,
        radial_kind: &'a RadialScalarKind,
    },
    /// Pre-computed analytic `(n_obs, n_centers, latent_dim)` jet.
    Jet(ArrayView3<'a, f64>),
}

/// Per-row latent coordinates `t ∈ ℝ^{N × d}` stored as a flat
/// row-major `Array1<f64>` of length `n_obs * latent_dim`.
#[derive(Debug, Clone)]
pub struct LatentCoordValues {
    /// Flattened (n_obs, latent_dim) latent matrix, row-major
    /// (so `values[n * d + k] = t_n[k]`).
    values: Array1<f64>,
    /// Number of rows `N`.
    n_obs: usize,
    /// Number of latent dimensions `d`.
    latent_dim: usize,
    /// Manifold used for per-row Riemannian updates.
    manifold: LatentManifold,
}

impl LatentCoordValues {
    /// Construct from a dense `(n_obs, latent_dim)` matrix.
    pub fn from_matrix(matrix: ArrayView2<'_, f64>) -> Self {
        Self::from_matrix_with_manifold(matrix, LatentManifold::Euclidean)
    }

    /// Construct from a dense matrix and explicit latent manifold.
    pub fn from_matrix_with_manifold(
        matrix: ArrayView2<'_, f64>,
        manifold: LatentManifold,
    ) -> Self {
        let n_obs = matrix.nrows();
        let latent_dim = matrix.ncols();
        let mut values = Array1::<f64>::zeros(n_obs * latent_dim);
        for n in 0..n_obs {
            for k in 0..latent_dim {
                values[n * latent_dim + k] = matrix[[n, k]];
            }
        }
        let mut out = Self {
            values,
            n_obs,
            latent_dim,
            manifold,
        };
        out.project_all_rows_to_manifold();
        out
    }

    /// Construct directly from a flat (`n_obs * latent_dim`) array.
    pub fn from_flat(values: Array1<f64>, n_obs: usize, latent_dim: usize) -> Self {
        Self::from_flat_with_manifold(values, n_obs, latent_dim, LatentManifold::Euclidean)
    }

    /// Construct directly from a flat array and explicit latent manifold.
    pub fn from_flat_with_manifold(
        values: Array1<f64>,
        n_obs: usize,
        latent_dim: usize,
        manifold: LatentManifold,
    ) -> Self {
        assert_eq!(
            values.len(),
            n_obs * latent_dim,
            "LatentCoordValues::from_flat: length {} != n_obs * latent_dim = {}",
            values.len(),
            n_obs * latent_dim
        );
        let mut out = Self {
            values,
            n_obs,
            latent_dim,
            manifold,
        };
        out.project_all_rows_to_manifold();
        out
    }

    pub fn n_obs(&self) -> usize {
        self.n_obs
    }

    pub fn latent_dim(&self) -> usize {
        self.latent_dim
    }

    /// Total length of the flat value array (= `n_obs * latent_dim`).
    pub fn len(&self) -> usize {
        self.values.len()
    }

    pub fn is_empty(&self) -> bool {
        self.values.is_empty()
    }

    pub fn manifold(&self) -> &LatentManifold {
        &self.manifold
    }

    pub fn with_manifold(&self, manifold: LatentManifold) -> Self {
        Self::from_flat_with_manifold(self.values.clone(), self.n_obs, self.latent_dim, manifold)
    }

    /// View the flat value array.
    pub fn as_flat(&self) -> &Array1<f64> {
        &self.values
    }

    /// View row `n` as a length-`d` slice.
    pub fn row(&self, n: usize) -> &[f64] {
        let start = n * self.latent_dim;
        let end = start + self.latent_dim;
        &self.values.as_slice().expect("contiguous")[start..end]
    }

    /// Materialize as a dense `(n_obs, latent_dim)` matrix view.
    /// Useful when handing `t` to a row-major basis evaluator
    /// (e.g. `build_duchon_basis`).
    pub fn as_matrix(&self) -> Array2<f64> {
        let mut out = Array2::<f64>::zeros((self.n_obs, self.latent_dim));
        for n in 0..self.n_obs {
            for k in 0..self.latent_dim {
                out[[n, k]] = self.values[n * self.latent_dim + k];
            }
        }
        out
    }

    /// Mutable write back of the flat value array, e.g. after a Newton step.
    pub fn set_flat(&mut self, flat: ArrayView1<'_, f64>) {
        assert_eq!(flat.len(), self.values.len());
        self.values.assign(&flat);
        self.project_all_rows_to_manifold();
    }

    /// Apply a flat tangent update row-by-row through the manifold retraction.
    pub fn retract_flat_delta(&mut self, delta: ArrayView1<'_, f64>) {
        assert_eq!(delta.len(), self.values.len());
        if self.manifold.is_euclidean() {
            for (t, dt) in self.values.iter_mut().zip(delta.iter()) {
                *t += *dt;
            }
            return;
        }
        assert_eq!(
            self.manifold.ambient_dim(self.latent_dim),
            self.latent_dim,
            "LatentCoordValues::retract_flat_delta: manifold ambient dim does not match latent_dim",
        );
        for n in 0..self.n_obs {
            let start = n * self.latent_dim;
            let end = start + self.latent_dim;
            let next = self.manifold.retract(
                self.values.slice(ndarray::s![start..end]),
                delta.slice(ndarray::s![start..end]),
            );
            for a in 0..self.latent_dim {
                self.values[start + a] = next[a];
            }
        }
    }

    fn project_all_rows_to_manifold(&mut self) {
        if self.manifold.is_euclidean() {
            return;
        }
        assert_eq!(self.manifold.ambient_dim(self.latent_dim), self.latent_dim);
        for n in 0..self.n_obs {
            let start = n * self.latent_dim;
            let end = start + self.latent_dim;
            let projected = self
                .manifold
                .project_point(self.values.slice(ndarray::s![start..end]));
            for a in 0..self.latent_dim {
                self.values[start + a] = projected[a];
            }
        }
    }

    /// Apply this latent block back to a `TermCollectionSpec`-style covariate
    /// table: returns the `(N, d)` materialized matrix that downstream basis
    /// evaluators (Duchon, Matérn, ...) take as their feature input.
    pub fn apply_tospec(&self) -> Array2<f64> {
        self.as_matrix()
    }

    /// Compute `∂Φ/∂t` for a radial-kernel design Φ — the original
    /// Duchon/Matérn path. See [`Self::design_gradient_wrt_t_dispatch`] for
    /// the basis-agnostic dispatch entry point.
    ///
    /// `centers` is `(n_centers, d)`.
    /// Returns a `(n_obs, n_centers, d)` jet whose `(n, k, a)` entry is
    /// `∂Φ_{n,k} / ∂t_{n,a} = q(r_{n,k}) · (t_{n,a} − c_{k,a})`.
    ///
    /// At `r = 0` the unit vector `(t − c)/r` is undefined; the radial scalar
    /// path therefore asks the kernel for the finite `q` limit and surfaces
    /// `BasisError::DegenerateAtCollision` when that limit does not exist.
    pub fn design_gradient_wrt_t(
        &self,
        centers: ArrayView2<'_, f64>,
        radial_kind: &RadialScalarKind,
    ) -> Result<Array3<f64>, BasisError> {
        let n_obs = self.n_obs;
        let d = self.latent_dim;
        let n_centers = centers.nrows();
        if centers.ncols() != d {
            crate::bail_dim_basis!(
                "LatentCoordValues::design_gradient_wrt_t center dimension mismatch: centers have {} cols but latent_dim is {}",
                centers.ncols(),
                d
            );
        }
        let mut jet = Array3::<f64>::zeros((n_obs, n_centers, d));
        for n in 0..n_obs {
            let t_n = self.row(n);
            for k in 0..n_centers {
                let mut r2 = 0.0_f64;
                for a in 0..d {
                    let delta = t_n[a] - centers[[k, a]];
                    r2 += delta * delta;
                }
                let r = r2.sqrt();
                let (_, q, _) = radial_kind.eval_design_triplet(r)?;
                if q == 0.0 {
                    continue;
                }
                for a in 0..d {
                    jet[[n, k, a]] = q * (t_n[a] - centers[[k, a]]);
                }
            }
        }
        Ok(jet)
    }

    /// Compute `∂Φ/∂t` for an arbitrary supported basis kind, by dispatching
    /// to the right closed-form chain rule.
    ///
    /// All radial-kernel bases (Duchon, Matérn) reduce to the same
    /// `q(r) · (t − c)` chain that `design_gradient_wrt_t` already implements.
    /// Non-radial bases (sphere, periodic-cyclic B-spline, tensor
    /// B-spline) carry their own analytic `(N, K, d)` jet — the caller
    /// pre-builds that jet using the matching `*_first_derivative_nd` helper
    /// in [`crate::basis`] and passes it in via
    /// [`InputLocationDerivative::Jet`].
    pub fn design_gradient_wrt_t_dispatch(
        &self,
        input: InputLocationDerivative<'_>,
    ) -> Result<Array3<f64>, BasisError> {
        match input {
            InputLocationDerivative::Radial {
                centers,
                radial_kind,
            } => self.design_gradient_wrt_t(centers, radial_kind),
            InputLocationDerivative::Jet(jet) => {
                if jet.shape() != [self.n_obs, jet.shape()[1], self.latent_dim] {
                    crate::bail_dim_basis!(
                        "LatentCoordValues::design_gradient_wrt_t_dispatch jet shape {:?} does not match latent shape ({}, {}, {})",
                        jet.shape(),
                        self.n_obs,
                        jet.shape()[1],
                        self.latent_dim
                    );
                }
                // The non-radial helpers already produce a (N, K, d) tensor
                // in the layout downstream contraction consumes. Return a copy
                // so the caller owns the data and is decoupled from the source
                // array's lifetime.
                Ok(jet.to_owned())
            }
        }
    }
}

fn wrap_to_period(x: f64, period: f64) -> f64 {
    assert!(
        period.is_finite() && period > 0.0,
        "wrap_to_period requires a finite positive period; got {period}"
    );
    let y = x.rem_euclid(period);
    if y == period { 0.0 } else { y }
}

/// Normalize `v[0..dim]` to a unit vector (for `LatentManifold::Sphere`
/// projection and retraction).
///
/// "Or axis": if the input is zero or non-finite (degenerate or numerical
/// mishap in caller), gracefully fall back to the canonical first axis
/// unit vector `[1, 0, …, 0]`. This removes a hard panic while preserving
/// the sphere contract that every returned point has unit Euclidean norm.
/// Callers (project_point / retract on Sphere) already ensure dim matches
/// the view length for the manifold component.
fn normalize_or_axis(v: ArrayView1<'_, f64>, dim: usize) -> Array1<f64> {
    let mut norm_sq = 0.0_f64;
    for a in 0..dim {
        norm_sq += v[a] * v[a];
    }
    const EPS: f64 = 1e-300; // protect against underflow/denorm that would give Inf
    if norm_sq > EPS && norm_sq.is_finite() {
        let inv = 1.0 / norm_sq.sqrt();
        let mut out = Array1::<f64>::zeros(dim);
        for a in 0..dim {
            out[a] = v[a] * inv;
        }
        out
    } else {
        // "or axis" fallback — beautiful, non-panicking resolution for
        // degenerate ambient vector on the sphere.
        let mut out = Array1::<f64>::zeros(dim);
        if dim > 0 {
            out[0] = 1.0;
        }
        out
    }
}

fn dot_views(a: ArrayView1<'_, f64>, b: ArrayView1<'_, f64>) -> f64 {
    assert_eq!(a.len(), b.len());
    let mut acc = 0.0_f64;
    for i in 0..a.len() {
        acc += a[i] * b[i];
    }
    acc
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::array;

    #[test]
    fn from_matrix_roundtrip() {
        let m = array![[1.0_f64, 2.0], [3.0, 4.0], [5.0, 6.0]];
        let lc = LatentCoordValues::from_matrix(m.view());
        assert_eq!(lc.n_obs(), 3);
        assert_eq!(lc.latent_dim(), 2);
        let back = lc.as_matrix();
        assert_eq!(back, m);
    }

    #[test]
    fn row_access() {
        let m = array![[1.0_f64, 2.0], [3.0, 4.0]];
        let lc = LatentCoordValues::from_matrix(m.view());
        assert_eq!(lc.row(0), &[1.0, 2.0]);
        assert_eq!(lc.row(1), &[3.0, 4.0]);
    }

    #[test]
    fn circle_manifold_update_wraps_into_canonical_interval() {
        let two_pi = std::f64::consts::TAU;
        let near_top = 6.2_f64;
        let m = array![[near_top]];
        let mut lc = LatentCoordValues::from_matrix_with_manifold(
            m.view(),
            LatentManifold::Circle { period: two_pi },
        );
        let delta = Array1::from(vec![1.5_f64]);
        lc.retract_flat_delta(delta.view());
        let updated = lc.row(0)[0];
        let expected = (near_top + 1.5).rem_euclid(two_pi);
        assert!(
            (0.0..two_pi).contains(&updated),
            "Circle retraction did not wrap into [0, 2π): got {updated}",
        );
        assert!(
            (updated - expected).abs() < 1e-12,
            "Circle retraction value mismatch: got {updated}, expected {expected}",
        );

        let large_delta = Array1::from(vec![10.0 * two_pi + 0.25_f64]);
        lc.retract_flat_delta(large_delta.view());
        let after_big = lc.row(0)[0];
        assert!(
            (0.0..two_pi).contains(&after_big),
            "Circle retraction did not wrap a large delta: got {after_big}",
        );
    }

    #[test]
    fn sphere_manifold_update_preserves_unit_norm() {
        let m = array![[1.0_f64, 0.0, 0.0]];
        let mut lc = LatentCoordValues::from_matrix_with_manifold(
            m.view(),
            LatentManifold::Sphere { dim: 3 },
        );
        let delta = Array1::from(vec![0.3_f64, 0.7, -0.2]);
        lc.retract_flat_delta(delta.view());
        let row = lc.row(0);
        let norm_sq: f64 = row.iter().map(|x| x * x).sum();
        assert!(
            (norm_sq.sqrt() - 1.0).abs() < 1e-12,
            "Sphere retraction did not preserve unit norm: ||t|| = {}",
            norm_sq.sqrt(),
        );

        let big_delta = Array1::from(vec![50.0_f64, -25.0, 13.0]);
        lc.retract_flat_delta(big_delta.view());
        let row2 = lc.row(0);
        let norm_sq2: f64 = row2.iter().map(|x| x * x).sum();
        assert!(
            (norm_sq2.sqrt() - 1.0).abs() < 1e-12,
            "Sphere retraction failed to renormalize after large delta: ||t|| = {}",
            norm_sq2.sqrt(),
        );
    }
}
