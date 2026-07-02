//! #2022 Workstream B — decoder-frame gauge quotient primitives for the `K = 1`
//! inner step.
//!
//! A single manifold atom contributes `exp(s_k)·Φ_k(t)·B_k` to the
//! reconstruction. Three continuous gauge freedoms make the raw
//! `(B_k, t, s_k)` parameterization non-identifiable, and every one of them
//! is the reason the terminal joint Hessian can be singular (residual gauge)
//! and the reason the historical joint path needed barrier / floor / keep-best
//! machinery to stay off the flat directions:
//!
//!  1. **SCALE.** `(B_k, s_k) ↦ (c·B_k, s_k − ln c)` leaves the contribution
//!     unchanged for any `c > 0`. Removed by pinning `‖B_k‖_F = 1` as a hard
//!     constraint and carrying the magnitude in the explicit log-amplitude
//!     `s_k` ([`retract_decoder_unit_frobenius`],
//!     [`unit_frobenius_tangent_projection`]). At `K = 1` the decoder-frame
//!     manifold is exactly the unit sphere `St(M·p, 1) = S^{M·p−1}` (the raw
//!     `vec(B_k)` normalized), so the "Stiefel constraint" of SAC_PLAN Part 3
//!     is the trivial `k = 1` sphere retraction: divide by the Frobenius norm.
//!  2. **CHART.** `t ↦ φ(t)` (reparameterization) leaves the decoded *curve*
//!     unchanged. Removed for `d = 1` by the unit-speed (arc-length) chart —
//!     already enforced in-loop by
//!     [`crate::chart_canonicalization::unit_speed_retraction`]; this module
//!     re-exports the sampling/gluing helpers built on top of it.
//!  3. **INTENSITY vs EXISTENCE.** the gate (existence) and the amplitude
//!     (intensity) were entangled while magnitude lived in `B_k`; the explicit
//!     `s_k` with the [`LogAmplitudeHoyerPrior`] (the #1939 "cone atom") is the
//!     sparse-amplitude prior on the *shape-normalized* atoms.
//!
//! Payoff, once the quotient is in force: the terminal joint evidence is
//! computed on the quotient (comparable normalizers across `K`), and the
//! same-manifold gluing test that SAC's birth race needs becomes the
//! two-parameter affine transition of the arc-length coordinate
//! ([`affine_chart_transition`]) — under unit-speed coordinates two atoms that
//! trace the same 1-manifold are related by `t_a = ±t_b + c` (slope exactly
//! `±1`), so stagewise arc-tiling is caught at birth.
//!
//! All derivatives here are hand-derived closed forms (SPEC: no autodiff
//! outside tests); the `#[cfg(test)]` module verifies each one against finite
//! differences, which SPEC permits *inside tests only*.

use ndarray::{Array1, Array2, ArrayView1, ArrayView2};

use super::{SaeBasisEvaluator, SaeManifoldAtom};

/// Frobenius norm `‖B‖_F = (Σ_{μ,j} B_{μj}²)^{1/2}` of a decoder block.
pub fn decoder_frobenius_norm(decoder: ArrayView2<'_, f64>) -> f64 {
    decoder.iter().map(|v| v * v).sum::<f64>().sqrt()
}

/// #2022 STEP 2 — pin `‖B_k‖_F = 1` as the hard SCALE-gauge constraint on one
/// atom's decoder frame, folding the removed magnitude into the explicit
/// log-amplitude so the contribution `exp(s_k)·Φ·B_k` is numerically UNCHANGED.
///
/// This is the `K = 1` decoder-frame retraction: `vec(B_k)` lives on the unit
/// sphere `S^{M·p−1} = St(M·p, 1)` and the retraction is the radial projection
/// `B_k ↦ B_k / ‖B_k‖_F`, `s_k ↦ s_k + ln‖B_k‖_F`. It is a genuine constraint
/// (not a heuristic normalization): after it the only decoder-frame freedom
/// left is the sphere itself (pure shape), and the scale ray has been quotiented
/// out into `s_k`. Idempotent — a frame already at unit norm is left untouched
/// and `false` is returned.
///
/// Delegates the byte-exact magnitude peel to
/// [`SaeManifoldAtom::absorb_decoder_norm_into_log_amplitude`] (which also keeps
/// the pullback-metric roughness Gram consistent). Returns `true` iff the frame
/// was rescaled (finite norm strictly off `1`).
pub fn retract_decoder_unit_frobenius(atom: &mut SaeManifoldAtom) -> bool {
    let norm = decoder_frobenius_norm(atom.decoder_coefficients.view());
    if !(norm.is_finite() && norm > 0.0) {
        return false;
    }
    if (norm - 1.0).abs() <= f64::EPSILON {
        return false;
    }
    atom.absorb_decoder_norm_into_log_amplitude(f64::MIN_POSITIVE);
    true
}

/// Project an ambient decoder gradient `G = ∂L/∂B_k` onto the tangent space of
/// the unit-Frobenius sphere at `B_k` (assumed `‖B_k‖_F = 1`): the SCALE
/// (radial) component is removed because it is carried by the log-amplitude
/// channel, not the frame.
///
/// The unit sphere `{B : ⟨B, B⟩_F = 1}` has tangent space `{Δ : ⟨Δ, B⟩_F = 0}`;
/// the metric projection is `Δ = G − ⟨G, B⟩_F · B`. This is the derivative
/// bookkeeping that keeps the frame step consistent with the retraction
/// [`retract_decoder_unit_frobenius`] (chain rule through the radial
/// projection): the along-`B` part of any raw gradient would only change the
/// magnitude, which `s_k` owns, so it is annihilated here. `B` need not be
/// exactly unit-norm — the projection uses `⟨G,B⟩/⟨B,B⟩` so it is correct for a
/// pre-retraction frame too.
pub fn unit_frobenius_tangent_projection(
    decoder: ArrayView2<'_, f64>,
    ambient_grad: ArrayView2<'_, f64>,
) -> Array2<f64> {
    let bb = decoder.iter().map(|v| v * v).sum::<f64>();
    let mut out = ambient_grad.to_owned();
    if !(bb > 0.0) {
        return out;
    }
    let gb: f64 = ambient_grad
        .iter()
        .zip(decoder.iter())
        .map(|(g, b)| g * b)
        .sum();
    let coeff = gb / bb;
    for (o, b) in out.iter_mut().zip(decoder.iter()) {
        *o -= coeff * b;
    }
    out
}

/// #1939 cone atom — the Hoyer sparsity prior on the atoms' explicit
/// amplitudes `a_k = exp(s_k)`, evaluated as an energy in the log-amplitudes
/// `s = (s_1, …, s_K)`.
///
/// With the SCALE gauge removed (every `B_k` unit-Frobenius, so `a_k` is the
/// atom's true intensity), a sparse dictionary is one where a few atoms carry
/// large amplitude and the rest are ~0. The Hoyer ratio
/// `‖a‖₁/‖a‖₂ ∈ [1, √K]` is the scale-invariant density of the amplitude
/// vector (`1` ⇔ one atom active, `√K` ⇔ all equal), so the prior toward
/// sparsity is the energy
///
/// ```text
///   E(s) = λ · ‖a‖₁ / ‖a‖₂,   a_k = exp(s_k).
/// ```
///
/// It is scale-invariant in `a` (adding a constant to every `s_k` leaves `E`
/// unchanged) — exactly the property the SCALE quotient demands — so it prices
/// the *distribution* of intensity across atoms, never the overall magnitude
/// (which the per-atom evidence owns). `λ` is a smoothing weight (REML/LAML
/// estimable like every other penalty coefficient), not a magic constant.
///
/// Writing `u_k = a_k / ‖a‖₂` (so `Σ u_k² = 1`) and `R = ‖a‖₁/‖a‖₂`, the exact
/// closed-form derivatives are
///
/// ```text
///   ∂E/∂s_k       = λ · u_k (1 − R u_k)
///   ∂²E/∂s_k∂s_j  = λ [ δ_kj u_k (1 − 2R u_k)
///                       − u_k u_j (u_j + u_k)
///                       + 3 R u_k² u_j² ]
/// ```
///
/// verified against finite differences in the test module.
#[derive(Debug, Clone)]
pub struct LogAmplitudeHoyerEnergy {
    /// Prior energy `E(s) = λ ‖a‖₁/‖a‖₂`.
    pub value: f64,
    /// Gradient `∂E/∂s_k`, length `K`.
    pub grad: Array1<f64>,
    /// Hessian `∂²E/∂s_k∂s_j`, shape `(K, K)`. Symmetric; may be indefinite (a
    /// ratio penalty is not convex in `s`), so a Newton assembly must PSD-majorize
    /// it before Cholesky exactly as the periodic ARD curvature is majorized.
    pub hess: Array2<f64>,
}

/// Evaluate the [`LogAmplitudeHoyerEnergy`] at log-amplitudes `s` with weight
/// `lambda`. Returns a zero energy (and zero derivatives) for `K ≤ 1` — the
/// Hoyer ratio is a constant `1` with a single atom, so it carries no gradient
/// and the whole prior is vacuous until there is more than one amplitude to
/// distribute mass across.
pub fn log_amplitude_hoyer_energy(s: ArrayView1<'_, f64>, lambda: f64) -> LogAmplitudeHoyerEnergy {
    let k = s.len();
    let mut grad = Array1::<f64>::zeros(k);
    let mut hess = Array2::<f64>::zeros((k, k));
    if k <= 1 {
        return LogAmplitudeHoyerEnergy {
            value: 0.0,
            grad,
            hess,
        };
    }
    // a_k = exp(s_k); shift by max(s) for overflow-free exponentials — E is
    // invariant to a common shift of s, so this is exact, not an approximation.
    let smax = s.iter().copied().fold(f64::NEG_INFINITY, f64::max);
    if !smax.is_finite() {
        return LogAmplitudeHoyerEnergy {
            value: 0.0,
            grad,
            hess,
        };
    }
    let a: Vec<f64> = s.iter().map(|&sk| (sk - smax).exp()).collect();
    let l1: f64 = a.iter().sum();
    let l2_sq: f64 = a.iter().map(|v| v * v).sum();
    let l2 = l2_sq.sqrt();
    if !(l2 > 0.0 && l1 > 0.0) {
        return LogAmplitudeHoyerEnergy {
            value: 0.0,
            grad,
            hess,
        };
    }
    let r = l1 / l2;
    let u: Vec<f64> = a.iter().map(|v| v / l2).collect();
    let value = lambda * r;
    for k1 in 0..k {
        grad[k1] = lambda * u[k1] * (1.0 - r * u[k1]);
    }
    for k1 in 0..k {
        for j in 0..k {
            let diag = if k1 == j {
                u[k1] * (1.0 - 2.0 * r * u[k1])
            } else {
                0.0
            };
            let cross = -u[k1] * u[j] * (u[j] + u[k1]) + 3.0 * r * u[k1] * u[k1] * u[j] * u[j];
            hess[[k1, j]] = lambda * (diag + cross);
        }
    }
    LogAmplitudeHoyerEnergy { value, grad, hess }
}

/// Sample one atom's decoded curve `γ(t) = exp(s)·Φ(t)·B` at the given latent
/// coordinates, returning the point set `(n × p)`. Pure forward evaluation (no
/// data, no refit) — the honest image the gluing test compares. `coords` is the
/// `d = 1` latent coordinate for each sample (e.g. a uniform arc-length grid
/// produced by
/// [`crate::chart_canonicalization::unit_speed_reparameterization`]).
pub fn sample_decoded_curve(
    evaluator: &dyn SaeBasisEvaluator,
    decoder: ArrayView2<'_, f64>,
    log_amplitude: f64,
    coords: ArrayView1<'_, f64>,
) -> Result<Array2<f64>, String> {
    let n = coords.len();
    let mut coords2 = Array2::<f64>::zeros((n, 1));
    for i in 0..n {
        coords2[[i, 0]] = coords[i];
    }
    let (phi, _jet) = evaluator.evaluate(coords2.view())?;
    if phi.ncols() != decoder.nrows() {
        return Err(format!(
            "sample_decoded_curve: basis width {} != decoder rows {}",
            phi.ncols(),
            decoder.nrows()
        ));
    }
    let mut pts = phi.dot(&decoder);
    if log_amplitude != 0.0 {
        let amp = log_amplitude.exp();
        pts.mapv_inplace(|v| v * amp);
    }
    Ok(pts)
}

/// The two-parameter affine transition `t_a ≈ slope·t_b + offset` relating the
/// arc-length coordinate of curve B to that of curve A, the object SAC's birth
/// race reads to decide whether a candidate atom lies on the SAME 1-manifold as
/// an existing atom.
#[derive(Debug, Clone)]
pub struct AffineChartTransition {
    /// Fitted slope. Under unit-speed (arc-length) coordinates a genuine
    /// same-manifold match forces `|slope| = 1` (orientation-preserving `+1`
    /// or reflected `−1`); the value is *fitted freely*, so `|slope|` near `1`
    /// is a verification, not an imposition.
    pub slope: f64,
    /// Fitted offset (the base-point shift `c` of `t_a = ±t_b + c`).
    pub offset: f64,
    /// RMS residual of the affine coordinate fit, in the same units as the
    /// arc-length coordinate. Small ⇔ the coordinate relation really is affine.
    pub coord_residual: f64,
    /// Mean nearest-point distance from curve B to curve A, normalized by the
    /// scale of curve A (its RMS radius about its centroid). Small ⇔ curve B
    /// geometrically lies ON curve A (period, tolerance-free).
    pub geometric_residual: f64,
}

impl AffineChartTransition {
    /// Same-manifold verdict at an explicit relative tolerance. Requires (i) the
    /// fitted slope to be within `rel_tol` of `±1` (arc-length rigidity), (ii)
    /// the affine coordinate residual to be within `rel_tol` of the coordinate
    /// scale `coord_scale` (the span of curve B's parameter), and (iii) the
    /// geometric residual within `rel_tol` (curve B lies on curve A).
    ///
    /// `rel_tol` is the caller's salience dial (SAC_PLAN Part 2: salience is a
    /// separate, explicit dial) — deliberately NOT hard-coded here, so this file
    /// carries no acceptance magic constant.
    pub fn same_manifold(&self, coord_scale: f64, rel_tol: f64) -> bool {
        let slope_ok = (self.slope.abs() - 1.0).abs() <= rel_tol;
        let coord_ok = coord_scale > 0.0 && self.coord_residual <= rel_tol * coord_scale;
        let geom_ok = self.geometric_residual <= rel_tol;
        slope_ok && coord_ok && geom_ok
    }
}

/// Fit the two-parameter affine transition between two arc-length-parameterized
/// curves. `points_a`/`points_b` are `(n_a × p)`/`(n_b × p)` point sets sampled
/// along the two decoded curves; `coords_a`/`coords_b` are their (arc-length)
/// latent coordinates. `period_a`, when `Some(P)`, unwraps the matched
/// `coord_a` sequence across the `S¹` branch cut so the regression is not
/// corrupted by the wrap (pass `None` for an interval/line chart).
///
/// Method (deterministic, closed-form, no autodiff): for each point of curve B
/// find its nearest point on curve A, giving a correspondence `(coord_b_j,
/// coord_a_j)` plus the point-to-point distance. Ordinary least squares on the
/// (branch-unwrapped) correspondences yields `slope`/`offset`; the RMS fit
/// residual is `coord_residual`; the mean matched distance normalized by curve
/// A's scale is `geometric_residual`.
pub fn affine_chart_transition(
    points_a: ArrayView2<'_, f64>,
    coords_a: ArrayView1<'_, f64>,
    points_b: ArrayView2<'_, f64>,
    coords_b: ArrayView1<'_, f64>,
    period_a: Option<f64>,
) -> Result<AffineChartTransition, String> {
    let (na, p) = points_a.dim();
    let (nb, pb) = points_b.dim();
    if p != pb {
        return Err(format!(
            "affine_chart_transition: output dims differ (a: {p}, b: {pb})"
        ));
    }
    if na != coords_a.len() || nb != coords_b.len() {
        return Err(format!(
            "affine_chart_transition: point/coord length mismatch (a: {na} vs {}, b: {nb} vs {})",
            coords_a.len(),
            coords_b.len()
        ));
    }
    if na < 2 || nb < 2 {
        return Err("affine_chart_transition: need at least two samples per curve".into());
    }

    // Curve A scale: RMS radius about its centroid, the normalizer for the
    // geometric residual (period-agnostic, tolerance-free).
    let mut centroid = vec![0.0_f64; p];
    for i in 0..na {
        for j in 0..p {
            centroid[j] += points_a[[i, j]];
        }
    }
    for c in centroid.iter_mut() {
        *c /= na as f64;
    }
    let mut scale_sq = 0.0_f64;
    for i in 0..na {
        for j in 0..p {
            let d = points_a[[i, j]] - centroid[j];
            scale_sq += d * d;
        }
    }
    let curve_scale = (scale_sq / na as f64).sqrt();

    // Nearest-A correspondence for every B point.
    let mut xs = Vec::with_capacity(nb); // coord_b
    let mut ys = Vec::with_capacity(nb); // coord_a of nearest A point
    let mut dist_sum = 0.0_f64;
    for jb in 0..nb {
        let mut best = f64::INFINITY;
        let mut best_i = 0usize;
        for ia in 0..na {
            let mut d = 0.0_f64;
            for c in 0..p {
                let diff = points_b[[jb, c]] - points_a[[ia, c]];
                d += diff * diff;
            }
            if d < best {
                best = d;
                best_i = ia;
            }
        }
        dist_sum += best.sqrt();
        xs.push(coords_b[jb]);
        ys.push(coords_a[best_i]);
    }
    let geometric_residual = if curve_scale > 0.0 {
        (dist_sum / nb as f64) / curve_scale
    } else {
        f64::INFINITY
    };

    // Order correspondences by coord_b and branch-unwrap coord_a so a circle
    // atom whose arc-length coordinate wraps modulo the period does not inject a
    // spurious `±P` jump into the regression.
    let mut order: Vec<usize> = (0..nb).collect();
    order.sort_by(|&i, &j| xs[i].partial_cmp(&xs[j]).unwrap_or(std::cmp::Ordering::Equal));
    let xo: Vec<f64> = order.iter().map(|&i| xs[i]).collect();
    let mut yo: Vec<f64> = order.iter().map(|&i| ys[i]).collect();
    if let Some(pp) = period_a {
        if pp > 0.0 {
            for idx in 1..yo.len() {
                let mut d = yo[idx] - yo[idx - 1];
                while d > 0.5 * pp {
                    yo[idx] -= pp;
                    d -= pp;
                }
                while d < -0.5 * pp {
                    yo[idx] += pp;
                    d += pp;
                }
            }
        }
    }

    // Ordinary least squares slope/offset on the unwrapped correspondences.
    let m = xo.len() as f64;
    let mean_x = xo.iter().sum::<f64>() / m;
    let mean_y = yo.iter().sum::<f64>() / m;
    let mut sxx = 0.0_f64;
    let mut sxy = 0.0_f64;
    for idx in 0..xo.len() {
        let dx = xo[idx] - mean_x;
        sxx += dx * dx;
        sxy += dx * (yo[idx] - mean_y);
    }
    if !(sxx > 0.0) {
        return Err(
            "affine_chart_transition: curve B coordinate has zero spread; slope undefined".into(),
        );
    }
    let slope = sxy / sxx;
    let offset = mean_y - slope * mean_x;
    let mut resid_sq = 0.0_f64;
    for idx in 0..xo.len() {
        let pred = slope * xo[idx] + offset;
        let e = yo[idx] - pred;
        resid_sq += e * e;
    }
    let coord_residual = (resid_sq / m).sqrt();

    Ok(AffineChartTransition {
        slope,
        offset,
        coord_residual,
        geometric_residual,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::{array, Array3};

    /// A trivial `d = 1` evaluator whose basis is the monomial patch
    /// `Φ(t) = [1, t]` — enough to build straight-line and circle-arc decoders
    /// for the gluing tests without pulling in the production evaluators.
    #[derive(Debug)]
    struct AffineLineEvaluator;

    impl SaeBasisEvaluator for AffineLineEvaluator {
        fn evaluate(
            &self,
            coords: ArrayView2<'_, f64>,
        ) -> Result<(Array2<f64>, Array3<f64>), String> {
            let n = coords.nrows();
            let mut phi = Array2::<f64>::zeros((n, 2));
            let mut jet = Array3::<f64>::zeros((n, 2, 1));
            for i in 0..n {
                let t = coords[[i, 0]];
                phi[[i, 0]] = 1.0;
                phi[[i, 1]] = t;
                jet[[i, 0, 0]] = 0.0;
                jet[[i, 1, 0]] = 1.0;
            }
            Ok((phi, jet))
        }

        fn second_jet_dyn(
            &self,
            _coords: ArrayView2<'_, f64>,
        ) -> Option<Result<ndarray::Array4<f64>, String>> {
            None
        }

        fn third_jet_dyn(
            &self,
            _coords: ArrayView2<'_, f64>,
        ) -> Option<Result<ndarray::Array5<f64>, String>> {
            None
        }
    }

    #[test]
    fn unit_frobenius_tangent_projection_kills_radial_component() {
        // B unit-Frobenius; the radial gradient c·B must project to ~0, and a
        // pure tangent gradient must pass through unchanged.
        let b = array![[0.6_f64, 0.0], [0.0, 0.8]]; // ‖B‖_F = 1
        let radial = b.mapv(|v| 2.5 * v);
        let proj = unit_frobenius_tangent_projection(b.view(), radial.view());
        let worst = proj.iter().fold(0.0_f64, |a, &v| a.max(v.abs()));
        assert!(worst < 1e-12, "radial gradient must project to 0, got {worst}");

        let tangent = array![[0.0_f64, 1.0], [-1.0, 0.0]]; // ⟨tangent, B⟩ = 0
        let proj_t = unit_frobenius_tangent_projection(b.view(), tangent.view());
        let drift = proj_t
            .iter()
            .zip(tangent.iter())
            .map(|(a, c)| (a - c).abs())
            .fold(0.0_f64, f64::max);
        assert!(drift < 1e-12, "tangent gradient must pass through, drift {drift}");
    }

    #[test]
    fn hoyer_energy_gradient_and_hessian_match_fd() {
        // Non-uniform amplitudes so every u_k is distinct and the derivatives are
        // genuinely exercised.
        let s = array![0.3_f64, -0.7, 1.1, 0.05];
        let lambda = 1.7_f64;
        let base = log_amplitude_hoyer_energy(s.view(), lambda);
        let h = 1e-6_f64;
        let k = s.len();
        // Gradient vs central difference of the value.
        for i in 0..k {
            let mut sp = s.clone();
            sp[i] += h;
            let mut sm = s.clone();
            sm[i] -= h;
            let vp = log_amplitude_hoyer_energy(sp.view(), lambda).value;
            let vm = log_amplitude_hoyer_energy(sm.view(), lambda).value;
            let fd = (vp - vm) / (2.0 * h);
            assert!(
                (base.grad[i] - fd).abs() <= 1e-6 * (1.0 + fd.abs()),
                "grad[{i}] {} != FD {fd}",
                base.grad[i]
            );
        }
        // Hessian vs central difference of the gradient.
        for i in 0..k {
            let mut sp = s.clone();
            sp[i] += h;
            let mut sm = s.clone();
            sm[i] -= h;
            let gp = log_amplitude_hoyer_energy(sp.view(), lambda).grad;
            let gm = log_amplitude_hoyer_energy(sm.view(), lambda).grad;
            for j in 0..k {
                let fd = (gp[j] - gm[j]) / (2.0 * h);
                assert!(
                    (base.hess[[j, i]] - fd).abs() <= 1e-5 * (1.0 + fd.abs()),
                    "hess[{j},{i}] {} != FD {fd}",
                    base.hess[[j, i]]
                );
            }
        }
        // Scale invariance: a common shift of s leaves E unchanged (SCALE gauge).
        let shifted = s.mapv(|v| v + 3.4);
        let e_shift = log_amplitude_hoyer_energy(shifted.view(), lambda).value;
        assert!(
            (e_shift - base.value).abs() <= 1e-9 * (1.0 + base.value.abs()),
            "Hoyer energy must be invariant to a common amplitude shift"
        );
    }

    #[test]
    fn hoyer_energy_prefers_sparse_over_dense() {
        // One dominant atom (sparse) must have LOWER energy than all-equal (dense).
        let sparse = array![2.0_f64, -3.0, -3.0, -3.0];
        let dense = array![0.0_f64, 0.0, 0.0, 0.0];
        let es = log_amplitude_hoyer_energy(sparse.view(), 1.0).value;
        let ed = log_amplitude_hoyer_energy(dense.view(), 1.0).value;
        assert!(es < ed, "sparse energy {es} must be below dense {ed}");
        // Dense K-vector realizes the ratio ceiling √K.
        assert!((ed - (4.0_f64).sqrt()).abs() < 1e-9, "dense ratio must be √K");
    }

    #[test]
    fn retract_decoder_unit_frobenius_is_image_frozen() {
        // Straight-line atom γ(t) = [1,t]·B with a non-unit decoder.
        let coords = array![[0.0_f64], [0.25], [0.5], [0.75], [1.0]];
        let ev = AffineLineEvaluator;
        let (phi, jet) = ev.evaluate(coords.view()).unwrap();
        let decoder = array![[2.0_f64, -1.0], [3.0, 0.5]]; // ‖B‖_F ≈ 3.775
        let atom = SaeManifoldAtom::new(
            "line",
            super::super::SaeAtomBasisKind::Linear,
            1,
            phi,
            jet,
            decoder.clone(),
            Array2::<f64>::eye(2),
        )
        .unwrap()
        .with_basis_evaluator(std::sync::Arc::new(AffineLineEvaluator));
        // Image before the retraction.
        let before = sample_decoded_curve(&ev, atom.decoder_coefficients.view(), atom.log_amplitude, coords.column(0))
            .unwrap();
        let mut atom = atom;
        let applied = retract_decoder_unit_frobenius(&mut atom);
        assert!(applied, "a non-unit decoder must be retracted");
        let norm = decoder_frobenius_norm(atom.decoder_coefficients.view());
        assert!((norm - 1.0).abs() < 1e-12, "‖B‖_F must be pinned to 1, got {norm}");
        // Image after — exp(s)·Φ·B must be byte-close to the original.
        let after = sample_decoded_curve(&ev, atom.decoder_coefficients.view(), atom.log_amplitude, coords.column(0))
            .unwrap();
        let drift = before
            .iter()
            .zip(after.iter())
            .map(|(a, b)| (a - b).abs())
            .fold(0.0_f64, f64::max);
        assert!(drift < 1e-10, "retraction must be image-frozen, drift {drift}");
        // Idempotent: a second retraction is a no-op.
        assert!(!retract_decoder_unit_frobenius(&mut atom), "retraction must be idempotent");
    }

    #[test]
    fn affine_transition_detects_same_line_with_reflection_and_offset() {
        // Curve A: straight segment through the origin, arc-length t ∈ [0, 1].
        // Curve B: the SAME line, reflected and offset — its arc-length coord is
        // t_a = -t_b + 1, i.e. slope -1, offset 1.
        let ev = AffineLineEvaluator;
        // Decoder makes γ(t) = [t·d] with unit-speed d (‖d‖ = 1) so t is arc length.
        let d = array![[0.0_f64, 0.0], [0.6, 0.8]]; // γ(t) = (0,0) + t·(0.6,0.8), speed 1
        let ca = Array1::linspace(0.0, 1.0, 11);
        let pts_a = sample_decoded_curve(&ev, d.view(), 0.0, ca.view()).unwrap();
        // B samples the same physical points but parameterized as t_b with
        // t_a = -t_b + 1  ⇒  physical point = (1 - t_b)·d.
        let cb = Array1::linspace(0.0, 1.0, 9);
        let db = array![[0.6_f64, 0.8], [-0.6, -0.8]]; // γ_b(t_b) = (0.6,0.8) + t_b·(-0.6,-0.8)
        let pts_b = sample_decoded_curve(&ev, db.view(), 0.0, cb.view()).unwrap();
        let tr = affine_chart_transition(pts_a.view(), ca.view(), pts_b.view(), cb.view(), None)
            .unwrap();
        assert!((tr.slope + 1.0).abs() < 1e-6, "slope must be -1, got {}", tr.slope);
        assert!((tr.offset - 1.0).abs() < 1e-6, "offset must be 1, got {}", tr.offset);
        assert!(tr.coord_residual < 1e-6, "coord residual {}", tr.coord_residual);
        assert!(tr.geometric_residual < 1e-6, "geometric residual {}", tr.geometric_residual);
        assert!(tr.same_manifold(1.0, 1e-3), "must be flagged same-manifold");
    }

    #[test]
    fn affine_transition_rejects_disjoint_curve() {
        // Curve B is a parallel line displaced far off curve A: the coordinate
        // regression may still fit a slope, but the GEOMETRIC residual is large,
        // so same_manifold must reject.
        let ev = AffineLineEvaluator;
        let da = array![[0.0_f64, 0.0], [1.0, 0.0]]; // A along x-axis
        let db = array![[0.0_f64, 5.0], [1.0, 0.0]]; // B parallel, y = 5 away
        let ca = Array1::linspace(0.0, 1.0, 11);
        let cb = Array1::linspace(0.0, 1.0, 11);
        let pts_a = sample_decoded_curve(&ev, da.view(), 0.0, ca.view()).unwrap();
        let pts_b = sample_decoded_curve(&ev, db.view(), 0.0, cb.view()).unwrap();
        let tr = affine_chart_transition(pts_a.view(), ca.view(), pts_b.view(), cb.view(), None)
            .unwrap();
        assert!(tr.geometric_residual > 1.0, "disjoint curve must have large geometric residual");
        assert!(!tr.same_manifold(1.0, 1e-2), "disjoint curve must be rejected");
    }
}
