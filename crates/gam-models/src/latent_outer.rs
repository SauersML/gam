//! The latent Gaussian REML outer problem: its objective and gradient in `t`, the manifold its trust region walks,
//! and the spectral seeds, moved from gam-pyffi (#2899 F6).

use gam_solve::gaussian_reml::{gaussian_reml_multi_closed_form_backward_from_fit, gaussian_reml_multi_closed_form_with_cache};
use gam_terms::AnalyticPenaltyRegistry;
use gam_terms::basis::input_loc_derivatives::contract_input_loc_gradient;
use gam_terms::basis::latent_design::build_latent_forward_design;
use gam_terms::latent::{aux_prior_targets, AuxPriorFamily, latent_analytic_penalty_value, latent_aux_prior_stats, latent_prior_score_and_aux_state_for_t, LatentAuxStrengthState, ValidatedDimSelectionPrecisions};
use ndarray::{Array1, Array2, ArrayView1, ArrayView2, ArrayView3};

/// Owned inputs for the latent outer-optimization objective.
///
/// Bundles the data the value/gradient evaluation needs so a single struct can
/// be reused across trust-region iterations and restarts without re-copying
/// from Python.
pub struct LatentOuterProblem {
    pub y: Array2<f64>,
    pub centers: Array2<f64>,
    pub penalty: Array2<f64>,
    pub weights: Option<Array1<f64>>,
    pub aux_u: Option<Array2<f64>>,
    pub dim_selection: Option<ValidatedDimSelectionPrecisions>,
    pub family: AuxPriorFamily,
    pub aux_strength: Option<f64>,
    pub init_lambda: Option<f64>,
    pub n_obs: usize,
    pub latent_dim: usize,
    pub m: usize,
    pub basis_kind: String,
    pub tensor_knots: Option<Array1<f64>>,
    pub tensor_knot_offsets: Option<Vec<usize>>,
    pub tensor_degrees: Option<Vec<usize>>,
    /// Per-axis chart period of the optimizer's manifold (radians) for the
    /// Duchon decoder; `None` on Euclidean / sphere so the decoder stays the
    /// open Euclidean basis. Derived from the `manifold` string in
    /// `gaussian_reml_optimize_latent` via `latent_manifold_periodic_descriptor`.
    pub periodic: Option<Vec<Option<f64>>>,
}

impl LatentOuterProblem {
    /// REML score (inner Gaussian REML plus aux/dim identifiability priors) and,
    /// when `want_grad`, the outer latent gradient `∂(reml_score)/∂t`.
    ///
    /// The value reproduces [`gaussian_reml_fit_latent`]'s `reml_score` and the
    /// gradient reproduces [`gaussian_reml_fit_latent_backward`]'s `grad_t` at
    /// `grad_reml_score = 1`, so the optimizer descends exactly the quantity the
    /// forward primitive reports. A non-finite or unsolvable configuration maps
    /// to `+∞` with no gradient, which the trust region rejects rather than
    /// propagating a NaN into the inner adjoint.
    pub fn value_and_grad(
        &self,
        t_flat: ArrayView1<'_, f64>,
        want_grad: bool,
    ) -> (f64, Option<Array1<f64>>) {
        match self.try_value_and_grad(t_flat, want_grad) {
            Ok(pair) => pair,
            Err(_) => (f64::INFINITY, None),
        }
    }

    pub fn try_value_and_grad(
        &self,
        t_flat: ArrayView1<'_, f64>,
        want_grad: bool,
    ) -> Result<(f64, Option<Array1<f64>>), String> {
        let (design, t_mat, jet) = build_latent_forward_design(
            &self.basis_kind,
            t_flat,
            self.n_obs,
            self.latent_dim,
            self.centers.view(),
            self.m,
            self.tensor_knots.as_ref().map(|a| a.view()),
            self.tensor_knot_offsets.as_deref(),
            self.tensor_degrees.as_deref(),
            self.periodic.as_deref(),
        )?;
        let weights_view = self.weights.as_ref().map(|w| w.view());
        let fit = gaussian_reml_multi_closed_form_with_cache(
            design.view(),
            self.y.view(),
            self.penalty.view(),
            weights_view,
            self.init_lambda,
            None,
        )
        .map_err(|err| err.to_string())?;
        let (prior_score, _aux_state) = latent_prior_score_and_aux_state_for_t(
            t_mat.view(),
            self.aux_u.as_ref().map(|a| a.view()),
            self.family,
            self.aux_strength,
            self.dim_selection.as_ref(),
        )?;
        let value = fit.reml_score + prior_score;
        if !value.is_finite() {
            return Ok((f64::INFINITY, None));
        }
        if !want_grad {
            return Ok((value, None));
        }
        let backward = gaussian_reml_multi_closed_form_backward_from_fit(
            design.view(),
            self.y.view(),
            self.penalty.view(),
            weights_view,
            &fit,
            0.0,
            None,
            None,
            1.0,
            0.0,
        )
        .map_err(|err| err.to_string())?;
        let mut grad_t = contract_input_loc_gradient(backward.grad_x.view(), &jet)
            .map_err(|err| err.to_string())?;
        // Identifiability-prior contributions, identical to the backward path's
        // grad_t assembly at `grad_reml_score = 1`.
        if let Some(u_arr) = self.aux_u.as_ref() {
            let u_view = u_arr.view();
            let stats =
                latent_aux_prior_stats(t_mat.view(), u_view, self.family, self.aux_strength)?;
            let residual = &t_mat - &stats.targets;
            let projected_residual = aux_prior_targets(residual.view(), u_view, self.family)?;
            let grad_base = residual - projected_residual;
            for n in 0..self.n_obs {
                for a in 0..self.latent_dim {
                    grad_t[n * self.latent_dim + a] += stats.strength.mu * grad_base[[n, a]];
                }
            }
        }
        if let Some(precisions) = self.dim_selection.as_ref() {
            for n in 0..self.n_obs {
                for a in 0..self.latent_dim {
                    let prec = precisions.physical()[a];
                    grad_t[n * self.latent_dim + a] += prec * t_mat[[n, a]];
                }
            }
        }
        if !grad_t.iter().all(|value| value.is_finite()) {
            return Ok((f64::INFINITY, None));
        }
        Ok((value, Some(grad_t)))
    }
}

/// Adapter exposing [`LatentOuterProblem`] to the Riemannian trust region.
pub struct LatentOuterObjective<'a> {
    pub problem: &'a LatentOuterProblem,
}

impl gam_geometry::RiemannianObjective for LatentOuterObjective<'_> {
    fn value_gradient(
        &mut self,
        point: ArrayView1<'_, f64>,
    ) -> gam_geometry::GeometryResult<(f64, Array1<f64>)> {
        // A degenerate point yields `+∞` and a zero gradient: the trust region
        // reads a zero gradient at the start as "stationary" (it stops at the
        // finite init) and a `+∞` trial value as a rejected step (it shrinks).
        match self.problem.value_and_grad(point, true) {
            (value, Some(grad)) => Ok((value, grad)),
            (_, None) => Ok((f64::INFINITY, Array1::<f64>::zeros(point.len()))),
        }
    }
}

/// Build the manifold the outer optimizer walks `t` on. `manifold` names the
/// per-observation geometry; the full latent lives on the `n_obs`-fold product.
/// Per-axis chart period for the latent decoder, derived from the optimizer's
/// manifold so the Duchon decoder is a genuine function ON that manifold.
///
/// The circle manifold (`src/geometry/circle.rs`) wraps each coordinate to
/// `[-π, π)`, i.e. period `2π = TAU` radians; the torus is its `d`-fold product.
/// The optimizer retracts the latent in radians on these charts, and the
/// periodic eigenmap seed (`latent_periodic_seed_start`) also produces radians,
/// so the decoder kernel distance must be measured modulo `TAU` per circular
/// axis and satisfy `Φ(θ) = Φ(θ + TAU)`. A non-periodic axis is `None`.
///
/// Euclidean / sphere return `None` (no axis is a circle): those latent fits
/// stay byte-identical to the open Euclidean Duchon basis. (`sphere` is `S^{d-1}`
/// embedded in `R^d` with NO periodic chart axis here — the spherical structure
/// is carried by the retraction, not by a per-axis wrap.)
pub fn latent_manifold_periodic_descriptor(
    manifold: &str,
    latent_dim: usize,
) -> Option<Vec<Option<f64>>> {
    match manifold.to_ascii_lowercase().replace('-', "_").as_str() {
        "circle" | "s1" if latent_dim == 1 => Some(vec![Some(std::f64::consts::TAU)]),
        "torus" => Some(vec![Some(std::f64::consts::TAU); latent_dim]),
        _ => None,
    }
}

pub fn build_latent_outer_manifold(
    manifold: &str,
    n_obs: usize,
    latent_dim: usize,
) -> Result<Box<dyn gam_geometry::RiemannianManifold>, String> {
    let per_point = match manifold.to_ascii_lowercase().replace('-', "_").as_str() {
        "euclidean" | "rn" => {
            // One flat Euclidean block over the whole latent is equivalent to
            // the product and avoids the per-observation slicing overhead.
            return Ok(Box::new(gam_geometry::EuclideanManifold::new(
                n_obs * latent_dim,
            )));
        }
        "circle" | "s1" => {
            if latent_dim != 1 {
                return Err(format!(
                    "circle latent manifold requires latent_dim == 1; got {latent_dim}"
                ));
            }
            gam_geometry::ManifoldSpec::Circle
        }
        "sphere" => {
            if latent_dim < 2 {
                return Err(format!(
                    "sphere latent manifold requires latent_dim >= 2 (S^{{d-1}} embeds in R^d); got {latent_dim}"
                ));
            }
            gam_geometry::ManifoldSpec::Sphere {
                intrinsic_dim: latent_dim - 1,
            }
        }
        "torus" => gam_geometry::ManifoldSpec::Torus { dim: latent_dim },
        other => {
            return Err(format!(
                "unknown latent manifold {other:?}; expected one of euclidean|circle|sphere|torus"
            ));
        }
    };
    let parts = std::iter::repeat_with(|| per_point.clone())
        .take(n_obs)
        .collect();
    gam_geometry::ManifoldSpec::Product(parts)
        .build()
        .map_err(|err| err.to_string())
}

/// Build the restart-0 start for the latent outer optimizer from a spectral
/// (Laplacian-eigenmaps) embedding of the responses `y`.
///
/// The embedding recovers the intrinsic coordinate up to monotone/rotation
/// gauge; each axis is then affinely mapped from `[0, 1]` onto the span of the
/// decoder `centers` for that axis so the seed lands where the basis `Φ` is
/// well-conditioned. On a *periodic* latent manifold (circle/torus) the natural
/// seed is the circular coordinate recovered from the leading Laplacian modes
/// (see [`latent_periodic_seed_start`]); the sphere has no closed-form spectral
/// seed here, so the caller's `t` is used unchanged.
///
/// A spread seed is essential, not optional: the outer optimizer's REML
/// objective is degenerate at a *collapsed* latent (all rows at the same
/// coordinate give identical decoder rows → a rank-deficient inner solve and no
/// usable descent direction). The default caller start is the all-zero vector,
/// which on a periodic manifold is exactly the collapsed configuration; without
/// a spread seed the circle/torus optimizer can never escape it (issue #876).
pub fn latent_spectral_seed_start(
    y: ArrayView2<'_, f64>,
    centers: ArrayView2<'_, f64>,
    manifold: &str,
    n_obs: usize,
    latent_dim: usize,
    seed_neighbors: usize,
    caller_t: ArrayView1<'_, f64>,
) -> Result<Array1<f64>, String> {
    let manifold_norm = manifold.to_ascii_lowercase().replace('-', "_");
    if matches!(manifold_norm.as_str(), "circle" | "s1" | "torus") {
        return latent_periodic_seed_start(y, n_obs, latent_dim, seed_neighbors, caller_t);
    }
    if !matches!(manifold_norm.as_str(), "euclidean" | "rn") {
        return Ok(caller_t.to_owned());
    }
    if y.nrows() != n_obs {
        return Err(format!(
            "spectral seed: y has {} rows but n_obs = {n_obs}",
            y.nrows()
        ));
    }
    // Too few rows to expose `latent_dim` non-trivial modes: fall back to the
    // caller's start rather than failing the whole optimize call.
    if n_obs < latent_dim + 2 {
        return Ok(caller_t.to_owned());
    }
    let coords = gam_geometry::laplacian_eigenmap_coords(y, latent_dim, seed_neighbors)?;
    // Per-axis target span from the decoder centers; fall back to [0, 1] when an
    // axis has no corresponding center column or a degenerate span.
    let mut start = Array1::<f64>::zeros(n_obs * latent_dim);
    for a in 0..latent_dim {
        let (lo, hi) = if a < centers.ncols() {
            let col = centers.column(a);
            let lo = col.iter().cloned().fold(f64::INFINITY, f64::min);
            let hi = col.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
            if lo.is_finite() && hi.is_finite() && hi > lo {
                (lo, hi)
            } else {
                (0.0, 1.0)
            }
        } else {
            (0.0, 1.0)
        };
        for n in 0..n_obs {
            start[n * latent_dim + a] = lo + coords[[n, a]] * (hi - lo);
        }
    }
    Ok(start)
}

/// Spectral seed for a *periodic* latent (circle / torus), returning each row's
/// angle in `[-π, π)` per axis.
///
/// On a circle the two leading non-trivial Laplacian-eigenmap modes of the
/// responses are (up to rotation/reflection — exactly the circle's gauge) the
/// `cos θ` / `sin θ` pair of the intrinsic angle, so `θ = atan2(sin-mode,
/// cos-mode)` recovers the circular coordinate directly. A torus of dimension
/// `d` is `d` independent circles; we recover one angle per axis from its own
/// pair of modes, requesting `2·d` modes from the embedding and pairing them in
/// order. The recovered angle is a *seed* — correct up to the periodic gauge the
/// decoder is free in — that the Riemannian outer optimizer then polishes.
///
/// Crucially this seed is *spread* around the circle, breaking the collapsed
/// all-zero start (issue #876). When there are too few rows to expose `2·d`
/// non-trivial modes the embedding cannot run; rather than start collapsed we
/// fall back to a deterministic equispaced angular sweep on each axis, which is
/// still non-degenerate (distinct decoder rows) so the optimizer has a usable
/// gradient.
pub fn latent_periodic_seed_start(
    y: ArrayView2<'_, f64>,
    n_obs: usize,
    latent_dim: usize,
    seed_neighbors: usize,
    caller_t: ArrayView1<'_, f64>,
) -> Result<Array1<f64>, String> {
    use std::f64::consts::TAU;

    if latent_dim == 0 {
        return Ok(caller_t.to_owned());
    }
    if y.nrows() != n_obs {
        return Err(format!(
            "periodic spectral seed: y has {} rows but n_obs = {n_obs}",
            y.nrows()
        ));
    }
    // A circle/torus axis needs two embedding modes (cos/sin); recover one angle
    // per axis. If the caller already supplied a *spread* warm start (not the
    // collapsed default), keep it — the optimizer can polish a good start, but it
    // can never escape a collapsed one. "Spread" is measured per axis by the
    // angular range the wrapped coordinates cover.
    let caller_spread = caller_t.len() == n_obs * latent_dim
        && (0..latent_dim).any(|a| {
            let mut lo = f64::INFINITY;
            let mut hi = f64::NEG_INFINITY;
            for n in 0..n_obs {
                let v = wrap_to_pi(caller_t[n * latent_dim + a]);
                lo = lo.min(v);
                hi = hi.max(v);
            }
            (hi - lo) > 1.0e-6
        });
    if caller_spread {
        return Ok(caller_t.to_owned());
    }

    let modes = 2 * latent_dim;
    // `laplacian_eigenmap_coords` needs `n >= modes + 2` rows to expose `modes`
    // non-trivial eigenvectors. With fewer rows, sweep angles deterministically.
    if n_obs < modes + 2 {
        let mut start = Array1::<f64>::zeros(n_obs * latent_dim);
        for a in 0..latent_dim {
            for n in 0..n_obs {
                let frac = if n_obs > 0 {
                    n as f64 / n_obs as f64
                } else {
                    0.0
                };
                start[n * latent_dim + a] = wrap_to_pi(frac * TAU);
            }
        }
        return Ok(start);
    }

    // The raw (un-rescaled) generalized eigenvectors are what carry the cos/sin
    // structure; `laplacian_eigenmap_coords` already rescales each axis to
    // [0, 1], which destroys the relative sign/scale needed for atan2. We
    // instead read `2·d` modes and undo the per-axis affine map by recentering
    // each mode to zero mean before pairing — the rescale is affine per mode, so
    // recentering recovers the angle up to the same rotation gauge.
    let coords = gam_geometry::laplacian_eigenmap_coords(y, modes, seed_neighbors)?;
    let mut mode_mean = vec![0.0f64; modes];
    for a in 0..modes {
        let mut sum = 0.0;
        for n in 0..n_obs {
            sum += coords[[n, a]];
        }
        mode_mean[a] = sum / n_obs as f64;
    }

    let mut start = Array1::<f64>::zeros(n_obs * latent_dim);
    for axis in 0..latent_dim {
        let cos_mode = 2 * axis;
        let sin_mode = 2 * axis + 1;
        for n in 0..n_obs {
            let c = coords[[n, cos_mode]] - mode_mean[cos_mode];
            let s = coords[[n, sin_mode]] - mode_mean[sin_mode];
            let angle = if c == 0.0 && s == 0.0 {
                // Degenerate row (both modes vanish): place it deterministically
                // around the circle so it does not coincide with its neighbours.
                wrap_to_pi((n as f64 / n_obs as f64) * TAU)
            } else {
                s.atan2(c)
            };
            start[n * latent_dim + axis] = angle;
        }
        // Guard against a collapsed axis (both modes constant → all angles
        // equal): fall back to an equispaced sweep on that axis only.
        let mut lo = f64::INFINITY;
        let mut hi = f64::NEG_INFINITY;
        for n in 0..n_obs {
            let v = start[n * latent_dim + axis];
            lo = lo.min(v);
            hi = hi.max(v);
        }
        if !(hi - lo > 1.0e-6) {
            for n in 0..n_obs {
                let frac = if n_obs > 0 {
                    n as f64 / n_obs as f64
                } else {
                    0.0
                };
                start[n * latent_dim + axis] = wrap_to_pi(frac * TAU);
            }
        }
    }
    Ok(start)
}

/// Wrap an angle to the half-open interval `[-π, π)`.
pub fn wrap_to_pi(angle: f64) -> f64 {
    use std::f64::consts::{PI, TAU};
    let mut a = angle % TAU;
    if a >= PI {
        a -= TAU;
    } else if a < -PI {
        a += TAU;
    }
    a
}

pub fn gaussian_reml_weight_vector_local(
    n_obs: usize,
    weights: Option<ArrayView1<'_, f64>>,
) -> Result<Array1<f64>, String> {
    match weights {
        Some(w) => {
            if w.len() != n_obs {
                return Err(format!(
                    "Gaussian REML weights length mismatch: expected {n_obs}, got {}",
                    w.len()
                ));
            }
            if w.iter().any(|value| !value.is_finite() || *value < 0.0) {
                return Err("Gaussian REML weights must be finite and non-negative".to_string());
            }
            Ok(w.to_owned())
        }
        None => Ok(Array1::ones(n_obs)),
    }
}

pub fn latent_scalar_weights_with_fisher(
    n_obs: usize,
    weights: Option<ArrayView1<'_, f64>>,
    fisher_w: Option<ArrayView3<'_, f64>>,
) -> Result<Option<Array1<f64>>, String> {
    let Some(fw) = fisher_w else {
        return Ok(weights.map(|w| w.to_owned()));
    };
    if fw.shape() != [n_obs, 1, 1] {
        return Err(format!(
            "fisher_w currently accepts scalar blocks of shape ({n_obs}, 1, 1) on this latent entry point; got {:?}",
            fw.shape()
        ));
    }
    let mut out = match weights {
        Some(w) => gaussian_reml_weight_vector_local(n_obs, Some(w))?,
        None => Array1::ones(n_obs),
    };
    for n in 0..n_obs {
        let v = fw[[n, 0, 0]];
        if !(v.is_finite() && v >= 0.0) {
            return Err(format!(
                "fisher_w[{n},0,0] must be finite and non-negative; got {v}"
            ));
        }
        out[n] *= v;
    }
    Ok(Some(out))
}

pub fn latent_row_weights(
    n_obs: usize,
    weights: Option<ArrayView1<'_, f64>>,
) -> Result<Array1<f64>, String> {
    match weights {
        Some(w) => gaussian_reml_weight_vector_local(n_obs, Some(w)),
        None => Ok(Array1::ones(n_obs)),
    }
}

pub fn validate_dense_fisher_w(
    n_obs: usize,
    n_outputs: usize,
    fisher_w: ArrayView3<'_, f64>,
) -> Result<(), String> {
    if fisher_w.shape() != [n_obs, n_outputs, n_outputs] {
        return Err(format!(
            "fisher_w dense blocks must have shape ({n_obs}, {n_outputs}, {n_outputs}); got {:?}",
            fisher_w.shape()
        ));
    }
    for n in 0..n_obs {
        for a in 0..n_outputs {
            for b in 0..n_outputs {
                let v = fisher_w[[n, a, b]];
                if !v.is_finite() {
                    return Err(format!("fisher_w[{n},{a},{b}] must be finite; got {v}"));
                }
            }
            if fisher_w[[n, a, a]] < 0.0 {
                return Err(format!(
                    "fisher_w[{n},{a},{a}] must be non-negative; got {}",
                    fisher_w[[n, a, a]]
                ));
            }
        }
    }
    Ok(())
}

pub fn gaussian_reml_fit_latent_impl(
    t_flat: ArrayView1<'_, f64>,
    y: ArrayView2<'_, f64>,
    n_obs: usize,
    latent_dim: usize,
    centers: ArrayView2<'_, f64>,
    m: usize,
    basis_kind: &str,
    tensor_knots_concat: Option<ArrayView1<'_, f64>>,
    tensor_knot_offsets: Option<&[usize]>,
    tensor_degrees: Option<&[usize]>,
    penalty: ArrayView2<'_, f64>,
    weights: Option<ArrayView1<'_, f64>>,
    init_lambda: Option<f64>,
    aux_u: Option<ArrayView2<'_, f64>>,
    aux_family: AuxPriorFamily,
    aux_strength: Option<f64>,
    dim_selection_precision: Option<&ValidatedDimSelectionPrecisions>,
    analytic_penalties: Option<&AnalyticPenaltyRegistry>,
    periodic: Option<&[Option<f64>]>,
) -> Result<
    (
        gam_solve::gaussian_reml::GaussianRemlMultiResult,
        Array2<f64>,
        Option<LatentAuxStrengthState>,
    ),
    String,
> {
    let (design, t_mat, _jet) = build_latent_forward_design(
        basis_kind,
        t_flat,
        n_obs,
        latent_dim,
        centers,
        m,
        tensor_knots_concat,
        tensor_knot_offsets,
        tensor_degrees,
        periodic,
    )?;
    // Build the (optionally) augmented Y/X stack carrying the identifiability
    // penalty. The penalty `½ μ ‖t − t_ref‖²` is *not* on the design Φ; it
    // acts on t directly. Because t enters Φ nonlinearly, we cannot fold it
    // into the inner Gaussian-closed-form solve without changing the solver.
    // We therefore evaluate the *penalty contribution* here and return it
    // for the caller to expose; the inner ridge stays unchanged.
    //
    // The forward path's responsibility is to produce a self-consistent fit
    // at the current t; the outer loop owns the gauge enforcement (it adds
    // ∂R_id/∂t to grad_t and walks t under that combined gradient).
    let mut fit = gaussian_reml_multi_closed_form_with_cache(
        design.view(),
        y,
        penalty,
        weights,
        init_lambda,
        None,
    )
    .map_err(|err| err.to_string())?;
    // Fixes audit-revised claim that ARD / aux-prior REML selection requires
    // normalized priors, not raw quadratic corrections alone.
    let (mut latent_prior_score, aux_strength_state) = latent_prior_score_and_aux_state_for_t(
        t_mat.view(),
        aux_u,
        aux_family,
        aux_strength,
        dim_selection_precision,
    )?;
    if let Some(registry) = analytic_penalties {
        latent_prior_score += latent_analytic_penalty_value(registry, t_flat)?;
    }
    fit.reml_score += latent_prior_score;
    Ok((fit, design, aux_strength_state))
}

#[cfg(test)]
mod latent_reml_tests {
    use super::*;
    use gam_terms::basis::latent_design::build_latent_duchon_design;
    use gam_terms::basis::latent_design::build_latent_forward_design;


    fn fixture(n: usize, dim: usize, k: usize, outputs: usize) -> (Array1<f64>, LatentOuterProblem) {
        let t = Array1::from_shape_fn(n * dim, |i| {
            0.5 + 0.38 * ((i + 1) as f64 * 1.731).sin()
        });
        let centers = Array2::from_shape_fn((k, dim), |(i, a)| {
            0.5 + 0.44 * ((i * dim + a + 1) as f64 * 2.317).sin()
        });
        let y = Array2::from_shape_fn((n, outputs), |(i, a)| {
            (3.0 * t[i * dim]).sin() + 0.3 * ((i + 1) as f64 * (a + 2) as f64 * 2.13).sin()
        });
        let problem = LatentOuterProblem {
            y,
            centers,
            penalty: Array2::eye(k),
            weights: Some(Array1::from_shape_fn(n, |i| 1.0 + 0.4 * (i as f64).cos())),
            aux_u: None,
            dim_selection: None,
            family: AuxPriorFamily::Ridge,
            aux_strength: None,
            init_lambda: None,
            n_obs: n,
            latent_dim: dim,
            m: 2,
            basis_kind: "duchon".to_string(),
            tensor_knots: None,
            tensor_knot_offsets: None,
            tensor_degrees: None,
            periodic: None,
        };
        (t, problem)
    }

    #[test]
    fn latent_reml_2833_design_jets_and_fixed_coefficient_frame() {
        for dim in [1, 2, 4] {
            let (t, problem) = fixture(16, dim, 9, 1);
            let evaluate = |point: ArrayView1<'_, f64>| {
                build_latent_forward_design(
                    "duchon", point, 16, dim, problem.centers.view(), 2,
                    None, None, None, None,
                ).unwrap()
            };
            let (design, _, jet) = evaluate(t.view());
            let h = 1e-6;
            for coordinate in 0..t.len() {
                let mut plus = t.clone();
                let mut minus = t.clone();
                plus[coordinate] += h;
                minus[coordinate] -= h;
                let (xp, _, _) = evaluate(plus.view());
                let (xm, _, _) = evaluate(minus.view());
                for row in 0..design.nrows() {
                    for col in 0..design.ncols() {
                        let expected = if row == coordinate / dim {
                            jet[[row, col, coordinate % dim]]
                        } else {
                            0.0
                        };
                        let observed = (xp[[row, col]] - xm[[row, col]]) / (2.0 * h);
                        assert!(
                            (observed - expected).abs() < 2e-6 * (1.0 + expected.abs()),
                            "dim={dim}, coordinate={coordinate}, row={row}, col={col}: \
                             derivative={expected}, finite difference={observed}"
                        );
                    }
                }
            }
        }
    }

    #[test]
    fn latent_reml_2833_reported_score_gradient_is_its_derivative() {
        for (n, dim, k, outputs) in [(12, 1, 5, 1), (20, 1, 8, 2), (24, 2, 9, 1)] {
            let (t, problem) = fixture(n, dim, k, outputs);
            let (value, gradient) = problem.try_value_and_grad(t.view(), true).unwrap();
            let gradient = gradient.unwrap();
            let h = 1e-5;
            let mut squared_error = 0.0;
            let mut squared_fd = 0.0;
            for coordinate in 0..t.len() {
                let mut plus = t.clone();
                let mut minus = t.clone();
                plus[coordinate] += h;
                minus[coordinate] -= h;
                let vp = problem.try_value_and_grad(plus.view(), false).unwrap().0;
                let vm = problem.try_value_and_grad(minus.view(), false).unwrap().0;
                let central = (vp - vm) / (2.0 * h);
                squared_error += (central - gradient[coordinate]).powi(2);
                squared_fd += central * central;
            }
            assert!(
                squared_error.sqrt() <= 1e-4 * (1.0 + squared_fd.sqrt()),
                "n={n}, dim={dim}, outputs={outputs}: error={}, fd norm={}",
                squared_error.sqrt(), squared_fd.sqrt()
            );
            let norm = gradient.dot(&gradient).sqrt();
            assert!(norm > 0.0);
            let next = &t - &gradient.mapv(|g| 1e-6 * g / norm);
            let next_value = problem.try_value_and_grad(next.view(), false).unwrap().0;
            assert!(next_value < value - 0.5e-6 * norm);
        }
    }

    /// Ordinary least squares R^2 of fitting `design @ beta ~= y` (single output),
    /// via an exact certified SPD solve of the normal equations. Used by the issue
    /// #876 latent-decoder regression below
    /// to quantify how well a candidate latent Duchon basis reconstructs a clean
    /// circular signal. NOT a tolerance to weaken — it is the recovery metric.
    fn latent_decoder_ols_r2(design: &Array2<f64>, y: &Array1<f64>) -> f64 {
        let gram = design.t().dot(design);
        let rhs = design.t().dot(y);
        let beta = gam_linalg::utils::certified_spd_factorize(
            &gram,
            "OLS normal equations for latent decoder recovery",
        )
        .expect("OLS normal-equations SPD factor")
        .solve(&rhs)
        .expect("OLS normal-equations solve for latent decoder recovery")
        .into_solution();
        let fitted = design.dot(&beta);
        let mean = y.sum() / y.len() as f64;
        let mut ss_res = 0.0;
        let mut ss_tot = 0.0;
        for (yi, fi) in y.iter().zip(fitted.iter()) {
            ss_res += (yi - fi) * (yi - fi);
            ss_tot += (yi - mean) * (yi - mean);
        }
        1.0 - ss_res / ss_tot
    }

    /// Issue #876: when the latent optimizer retracts on a PERIODIC manifold
    /// (circle, radians wrapped to [-pi, pi), period TAU), the latent Duchon decoder
    /// MUST be a function ON the circle. The pre-fix open Euclidean basis violated
    /// `Phi(theta) = Phi(theta + TAU)` and measured the kernel distance across the
    /// seam (theta = pi - eps vs -pi + eps, adjacent on the circle) as ~2*pi apart.
    ///
    /// This pins both halves of the fix:
    ///   (a) seam consistency: the PERIODIC basis satisfies `Phi(theta) ~= Phi(theta
    ///       + TAU)` row-for-row, while the OPEN basis does not;
    ///   (b) recovery: on a clean circular signal `y = cos(theta) + 0.5 sin(2 theta)`
    ///       sampled across the WHOLE circle (including the seam), the periodic
    ///       decoder reconstructs the angle structure (high R^2), whereas the open
    ///       decoder is materially worse near the seam.
    ///
    /// `latent_manifold_periodic_descriptor("circle", 1)` is the exact descriptor the
    /// optimizer feeds the decoder, so this exercises the production path.
    #[test]
    fn issue_876_periodic_latent_duchon_decoder_is_seam_consistent_and_recovers_circle() {
        let tau = std::f64::consts::TAU;
        let latent_dim = 1usize;
        let m = 2usize;

        // Radian centers spanning the circle, matching the optimizer's chart and the
        // periodic eigenmap seed (`latent_periodic_seed_start`), which both live in
        // [-pi, pi). Deliberately include centers near both seam edges.
        let n_centers = 12usize;
        let mut centers = Array2::<f64>::zeros((n_centers, latent_dim));
        for k in 0..n_centers {
            // Evenly spaced angles in [-pi, pi).
            centers[[k, 0]] = -std::f64::consts::PI + tau * (k as f64) / (n_centers as f64);
        }

        // The descriptor the production optimizer derives for a 1-D circle chart.
        let descriptor = latent_manifold_periodic_descriptor("circle", latent_dim)
            .expect("circle manifold must yield a periodic descriptor");
        assert_eq!(descriptor, vec![Some(tau)]);

        // (a) Seam consistency. Build the design at sample angles theta and again at
        // theta + TAU (same point on the circle). For the periodic decoder the two
        // designs must agree row-for-row; the open Euclidean decoder must not.
        let n_obs = 40usize;
        let theta: Vec<f64> = (0..n_obs)
            .map(|i| -std::f64::consts::PI + tau * (i as f64 + 0.5) / (n_obs as f64))
            .collect();
        let theta_shift: Vec<f64> = theta.iter().map(|&a| a + tau).collect();

        let t_flat = Array1::from(theta.clone());
        let t_flat_shift = Array1::from(theta_shift.clone());

        let (design_per, _) = build_latent_duchon_design(
            t_flat.view(),
            n_obs,
            latent_dim,
            centers.view(),
            m,
            Some(descriptor.as_slice()),
        )
        .expect("periodic latent Duchon design");
        let (design_per_shift, _) = build_latent_duchon_design(
            t_flat_shift.view(),
            n_obs,
            latent_dim,
            centers.view(),
            m,
            Some(descriptor.as_slice()),
        )
        .expect("periodic latent Duchon design (shifted by TAU)");

        assert_eq!(design_per.dim(), design_per_shift.dim());
        let mut max_per_seam_gap = 0.0_f64;
        for (a, b) in design_per.iter().zip(design_per_shift.iter()) {
            max_per_seam_gap = max_per_seam_gap.max((a - b).abs());
        }
        // Periodic decoder is a genuine function on the circle: Phi(theta) = Phi(theta + TAU).
        assert!(
            max_per_seam_gap <= 1.0e-8,
            "periodic latent decoder must satisfy Phi(theta) = Phi(theta + TAU); \
             max row gap was {max_per_seam_gap}"
        );

        // The OPEN Euclidean decoder (None) is NOT periodic: it must visibly differ.
        let (design_open, _) =
            build_latent_duchon_design(t_flat.view(), n_obs, latent_dim, centers.view(), m, None)
                .expect("open Euclidean latent Duchon design");
        let (design_open_shift, _) = build_latent_duchon_design(
            t_flat_shift.view(),
            n_obs,
            latent_dim,
            centers.view(),
            m,
            None,
        )
        .expect("open Euclidean latent Duchon design (shifted by TAU)");
        let mut max_open_seam_gap = 0.0_f64;
        for (a, b) in design_open.iter().zip(design_open_shift.iter()) {
            max_open_seam_gap = max_open_seam_gap.max((a - b).abs());
        }
        assert!(
            max_open_seam_gap > 1.0e-3,
            "open Euclidean latent decoder must NOT be periodic (control); \
             max row gap was {max_open_seam_gap}"
        );

        // (b) Recovery on a clean circular signal across the whole circle, including
        // the seam. The truth is a genuine function on the circle.
        let y: Array1<f64> = Array1::from_iter(theta.iter().map(|&a| a.cos() + 0.5 * (2.0 * a).sin()));

        let r2_periodic = latent_decoder_ols_r2(&design_per, &y);
        let r2_open = latent_decoder_ols_r2(&design_open, &y);

        // The periodic decoder recovers the angle structure (not collapsed).
        assert!(
            r2_periodic >= 0.95,
            "periodic latent decoder must recover the circular signal; R^2 = {r2_periodic}"
        );

        // Compare reconstruction error specifically at the seam-adjacent points (the
        // first and last sample angles, which straddle theta = +/- pi). The periodic
        // decoder must not have the cross-seam discontinuity the open basis carries.
        let solve_fitted = |design: &Array2<f64>| -> Array1<f64> {
            let gram = design.t().dot(design);
            let rhs = design.t().dot(&y);
            let beta = gam_linalg::utils::certified_spd_factorize(&gram, "seam OLS normal equations")
                .expect("seam OLS SPD factor")
                .solve(&rhs)
                .expect("seam OLS solve")
                .into_solution();
            design.dot(&beta)
        };
        let fitted_per = solve_fitted(&design_per);
        let fitted_open = solve_fitted(&design_open);
        // Seam-straddling pair: last sample (just below +pi) and first (just above -pi).
        let seam_err = |fitted: &Array1<f64>| -> f64 {
            let e_first = (fitted[0] - y[0]).abs();
            let e_last = (fitted[n_obs - 1] - y[n_obs - 1]).abs();
            e_first.max(e_last)
        };
        assert!(
            seam_err(&fitted_per) <= seam_err(&fitted_open) + 1.0e-9,
            "periodic decoder must reconstruct seam-adjacent points at least as well \
             as the open decoder: periodic seam err = {}, open seam err = {}, R^2 open = {r2_open}",
            seam_err(&fitted_per),
            seam_err(&fitted_open),
        );
    }
}
