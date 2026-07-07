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

use super::{SaeBasisEvaluator, SaeManifoldAtom, SaeManifoldRho, SaeManifoldTerm, Side};
use gam_linalg::faer_ndarray::FaerCholesky;

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
    order.sort_by(|&i, &j| {
        xs[i]
            .partial_cmp(&xs[j])
            .unwrap_or(std::cmp::Ordering::Equal)
    });
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

impl SaeManifoldTerm {
    /// #2022 STEP 2 — in-loop decoder-frame gauge retraction: pin `‖B_k‖_F = 1`
    /// on every atom, folding each removed magnitude into that atom's explicit
    /// log-amplitude. The companion of
    /// [`Self::retract_unit_speed_charts_in_loop`] for the SCALE gauge: both are
    /// IMAGE-FROZEN (the decoded contribution `exp(s)·Φ·B` is numerically
    /// unchanged), so the data-fit, smoothness, and terminal Laplace evidence
    /// are invariant — only the `(B_k, s_k)` representation moves onto the
    /// quotient (unit-Frobenius frame + explicit amplitude).
    ///
    /// Cadence (identical to the unit-speed chart retraction): call at a
    /// post-acceptance chart-refresh boundary, NEVER inside a line search. Within
    /// one inner solve the border `B_k` is free to carry scale; peeling it into
    /// `s_k` here between solves is what makes the *converged* dictionary sit on
    /// the SCALE quotient — so terminal evidence normalizers are comparable
    /// across `K` and the decoder-norm collapse guards (which key on `‖B_k‖`)
    /// become inert, their dead-atom signal migrating to the amplitude `s_k`.
    ///
    /// Returns the number of atoms whose frame was rescaled (a strict no-op —
    /// `0` returned — once every atom is already unit-Frobenius, so it is
    /// idempotent at a boundary).
    pub fn retract_decoder_gauge_in_loop(&mut self) -> usize {
        let mut retracted = 0usize;
        for atom in self.atoms.iter_mut() {
            if retract_decoder_unit_frobenius(atom) {
                retracted += 1;
            }
        }
        retracted
    }

    /// #1939 Design B — SCOPED cone-atom retraction: unit-Frobenius-retract ONLY
    /// the atoms whose decoder has COLLAPSED relative to its dictionary peers,
    /// leaving healthy atoms' scale in `B` untouched. Returns the number retracted.
    ///
    /// The unconditional [`Self::retract_decoder_gauge_in_loop`] forces `‖B_k‖≡1`
    /// on EVERY atom every accepted iterate, which fights every subsystem that
    /// assumes scale lives in `B` (the isometry `‖B‖⁴` pullback #673/#795, the
    /// `‖B‖`-keyed norm guards, the β-Newton magnitude step). On a HEALTHY fit that
    /// destabilizes the trajectory — empirically it DETONATES a healthy K=2 disjoint
    /// fit (EV → −1e128, a runaway β step after the forced unit retraction). So
    /// retracting healthy atoms is the harm; retracting a genuinely-collapsed atom is
    /// the cure (it re-homes the vanished decoder onto the unit sphere so the paired
    /// amplitude solve can restore its magnitude — the born-atom `0.7255→0.0023`
    /// recovery).
    ///
    /// The collapse trigger REUSES the existing decoder-norm guard threshold
    /// (`SAE_ATOM_DECODER_NORM_COLLAPSE_RATIO · median‖B‖`, assignment.rs — an
    /// existing DERIVED constant, no new magic number) so an atom is retracted iff
    /// its decoder is below the same bar `enforce_decoder_norm_guard` calls a breach.
    /// Two floors keep it safe: (i) `k < 2` returns 0 — a lone atom has no peer to be
    /// "collapsed" relative to, and a K=1 low-amplitude decoder is HEALTHY (its scale
    /// legitimately lives in `B`), so it is never retracted (this is what makes the
    /// K=1 low-amp crash structurally impossible); (ii) an atom whose `‖B‖` is below
    /// a machine floor is skipped — it carries no direction to normalize (`B/‖B‖` is
    /// undefined), it needs reseeding not retraction (seed-fix's lane).
    pub fn retract_collapsed_decoders_in_loop(&mut self) -> usize {
        let k = self.k_atoms();
        // A single atom has no dictionary peer to be collapsed *relative to*, and a
        // K=1 low-amplitude decoder is healthy — never retract it (mirrors the
        // K<2 early-out in `enforce_decoder_norm_guard`).
        if k < 2 {
            return 0;
        }
        let norms: Vec<f64> = self
            .atoms
            .iter()
            .map(|atom| atom.contribution_frobenius_scale())
            .collect();
        // "Healthy dictionary scale" to measure collapse against = the MAX decoder
        // scale (definitionally a surviving atom). NOT the median: the median is the
        // guard's statistic but it DEGENERATES whenever the MAJORITY of atoms
        // collapse, because the collapsed atoms THEMSELVES set the median. At K=3
        // with two co-vanished atoms (real IBP data: ‖B‖=[2.6, ~0, ~0]) the median
        // is ~0 — either exactly 0 (a median-keyed floor finds no reference) or
        // tiny-nonzero (a median-keyed `1e-3·median` floor sits BELOW the collapsed
        // atoms, so they aren't flagged). Both miss exactly the failure this exists
        // to fix. Keying on the max survivor is robust to any collapse fraction and
        // leaves healthy fits unchanged (the largest atom's 1e-3·max floor never
        // flags a peer of ordinary magnitude). `median` is kept for the diagnostic
        // trace only. Genuine TOTAL collapse (max 0) has no scale — return 0 and
        // defer to the reseed arm.
        let mut sorted = norms.clone();
        sorted.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
        let median = if k % 2 == 1 {
            sorted[k / 2]
        } else {
            0.5 * (sorted[k / 2 - 1] + sorted[k / 2])
        };
        let reference = norms.iter().copied().fold(0.0_f64, f64::max);
        if !(reference > 0.0) {
            return 0;
        }
        let breach_floor = crate::assignment::SAE_ATOM_DECODER_NORM_COLLAPSE_RATIO * reference;
        // Below this a decoder carries no usable direction to normalize; retracting
        // `B/‖B‖` would amplify pure round-off. Machine-scaled to the dictionary.
        let direction_floor = 1.0e-12 * reference;
        let mut retracted = 0usize;
        // A decoder BELOW the direction floor is collapsed but UN-RETRACTABLE (no
        // direction to normalize) — it needs a fresh-direction reseed, not this
        // retraction. Count it so a null wheel A/B is self-diagnosing (fired-but-
        // didn't-help vs never-fired-because-literal-zero — see the trace below).
        let mut unretractable_zero = 0usize;
        for idx in 0..k {
            if norms[idx] < breach_floor {
                if norms[idx] > direction_floor {
                    if retract_decoder_unit_frobenius(&mut self.atoms[idx]) {
                        retracted += 1;
                    }
                } else {
                    unretractable_zero += 1;
                }
            }
        }
        // #1939 wheel diagnostic (opt-in path only — this runs under the
        // `cone_atom_recovery` or `quotient_scale` (#2100) breach-gated boundary
        // retraction). Emit the RAW UNROUNDED per-atom norms, the
        // breach/direction thresholds, and the fired / literal-zero counts, so the
        // near-zero-vs-literal-zero question is answered inside the A/B run: a
        // collapse at `1e-3` is retracted (`retracted>0`), a collapse at `1e-16` is
        // an `unretractable_zero` (Design B correctly no-ops → needs a backfit
        // reseed, not this). Under a healthy dictionary nothing breaches and this is
        // a bit-for-bit no-op with a benign one-line trace. The printed `norms` are
        // physical contribution scales `exp(s_k)‖B_k‖_F`, so the diagnostic keeps
        // its meaning under the scale quotient.
        let norms_fmt: Vec<String> = norms.iter().map(|v| format!("{v:.6e}")).collect();
        log::warn!(
            "[#1939 cone-atom] k={k} norms=[{}] median={median:.6e} reference={reference:.6e} \
             breach_floor={breach_floor:.6e} direction_floor={direction_floor:.6e} \
             retracted={retracted} unretractable_zero={unretractable_zero}",
            norms_fmt.join(", ")
        );
        retracted
    }

    /// #1939 cone-atom amplitude solve — the OTHER half of the scale quotient.
    /// After [`Self::retract_decoder_gauge_in_loop`] pins every `‖B_k‖_F = 1`, set
    /// each atom's explicit log-amplitude `s_k` to the RECONSTRUCTION-OPTIMAL
    /// magnitude by a small non-negative least squares over the amplitudes
    /// `β = exp(s)`: with the frozen unit-`B` gated designs
    /// `D_k = diag(a_{·k})·Φ_k·B_k` (the atom's reconstruction at unit amplitude),
    /// `β` minimises the fit's OWN weighted reconstruction data-fit
    /// `0.5·Σ_i w_i·‖W_i(target_i − Σ_k β_k D_{k,i})‖²` — the same per-row RowMetric
    /// whitening `W_i` and #991 design-honesty weights `w_i` that
    /// [`Self::data_fit_for_reconstruction`] / `loss_scaled` use — as a `K×K` SPD
    /// weighted normal-equation solve. The weighting is DELEGATED to the fit's own
    /// `whiten_residual_row`, never re-derived in raw space: a parallel raw-space
    /// LSQ diverges from the fit's objective under a whitening / row-weighted
    /// metric and regresses those fits (the −151 / whitened_k2 class). Under a
    /// Euclidean, unweighted metric it is bit-for-bit the plain `‖target − Σβ_kD_k‖²`.
    ///
    /// This is load-bearing: the retraction alone only FOLDS `‖B‖` into `s`
    /// (magnitude-neutral), so `s` then DRIFTS to collapse under the penalty's
    /// residual shape gradient — the scale-collapse merely relocates from `B` to
    /// `s`. Pinning `s` to the data optimum here is what keeps a low-signal atom's
    /// amplitude at its true (small) value with a HEALTHY unit-norm decoder shape,
    /// instead of co-vanishing (the K≥2 born-atom `0.7255→0.0023` decoder collapse
    /// / the #1026 degenerate-amplitude thrash). Idempotent at a fixed point.
    pub fn optimize_log_amplitudes_closed_form(
        &mut self,
        target: ArrayView2<'_, f64>,
        rho: &SaeManifoldRho,
    ) -> Result<(), String> {
        let k = self.k_atoms();
        if k == 0 {
            return Ok(());
        }
        // Per-atom UNIT-amplitude contribution C_k, taken from the term's OWN
        // `try_fitted_for_rho` reconstruction so it matches the fitted output the
        // data-fit measures exactly (a naive `Φ_k·B_k·gate` mis-matches the gated
        // `exp(s)·Φ·B` fitted output and sets a wrong amplitude). The fitted output
        // is LINEAR in `exp(s_k)`
        // (`fitted = Σ_k a_k·exp(s_k)·Φ_k·B_k`, and the gate `a_k` is amplitude-
        // independent), so toggling atom `k` on (`s=0`) and the rest off
        // (`exp(s)→0`) reads off `C_k`.
        let saved: Vec<f64> = self.atoms.iter().map(|a| a.log_amplitude).collect();
        // Monotone safeguard: the amplitude LSQ is optimal for the frozen unit-B
        // designs, but the retraction↔amplitude↔Newton alternation is not
        // guaranteed contractive, so the update can make the reconstruction WORSE.
        // Bank the pre-solve data-fit and revert if the update degrades it
        // (keep-best, exactly like the #1026 incumbent restore), so the cone-atom
        // can never regress a fit that was recovering. CRITICAL (team-lead
        // #1939): the banked objective is the fit's OWN weighted reconstruction
        // data-fit `data_fit_for_reconstruction` (per-row RowMetric whitening +
        // #991 design-honesty row weights), NOT a raw unweighted EV — a raw EV is
        // a parallel re-derivation that diverges from what the fit minimises under
        // a whitening / row-weighted metric and mis-fires the guard (the −151 /
        // whitened_k2 regression class). The retraction is magnitude-neutral, so
        // this reconstruction is the pre-retraction one. Lower data-fit is better.
        let df_before = self
            .try_fitted_for_rho(rho)
            .and_then(|recon| self.data_fit_for_reconstruction(target, recon.view()))
            .unwrap_or(f64::INFINITY);
        const OFF: f64 = -700.0; // exp(-700) underflows to 0: contribution off.
        let mut designs: Vec<Array2<f64>> = Vec::with_capacity(k);
        let mut probe_err: Option<String> = None;
        for kk in 0..k {
            for (j, atom) in self.atoms.iter_mut().enumerate() {
                atom.log_amplitude = if j == kk { 0.0 } else { OFF };
            }
            match self.try_fitted_for_rho(rho) {
                Ok(c) => designs.push(c),
                Err(e) => {
                    probe_err = Some(e);
                    break;
                }
            }
        }
        for (j, atom) in self.atoms.iter_mut().enumerate() {
            atom.log_amplitude = saved[j];
        }
        if let Some(e) = probe_err {
            return Err(format!(
                "optimize_log_amplitudes_closed_form: per-atom probe fit failed: {e}"
            ));
        }
        // Normal equations for min_β 0.5·Σ_i w_i·‖W_i(target_i − Σ_k β_k C_{k,i})‖²,
        // β = exp(s) ≥ 0 — the WEIGHTED reconstruction VarPro profile, i.e. exactly
        // the objective `data_fit_for_reconstruction` (hence `loss_scaled`)
        // minimises. `W_i` is the per-row RowMetric whitening factor `U_iᵀ` and
        // `w_i` the #991 design-honesty row weight. Rather than re-derive the
        // weighting (the −151 raw-space mismatch class), whiten each per-atom
        // design row and the target row through the fit's OWN `whiten_residual_row`
        // (linear in its argument, `‖W_i r‖² = rᵀ W_i r`; identity/rank-p for a
        // Euclidean metric, so this reduces to the plain Gram bit-for-bit when no
        // metric/row-weight is installed) and scale by `√w_i`. The `0.5` and the
        // constant `√w_i`/`W_i` factors are shared by Gram and RHS, so the profile
        // minimiser is unchanged; forming them in the whitened rank space makes the
        // Gram the true weighted `CᵀWC`.
        let metric = self.row_metric();
        let whitens = metric.is_some_and(|m| m.whitens_likelihood());
        let row_loss_w = self.row_loss_weights();
        let n = target.nrows();
        let mut gram = Array2::<f64>::zeros((k, k));
        let mut rhs = Array1::<f64>::zeros(k);
        // Per-row whitened design/target rows reused across the K(K+1)/2 + K dot
        // products (K small); rank-length under a factored metric, p under None.
        let mut wdesign: Vec<Vec<f64>> = vec![Vec::new(); k];
        for row in 0..n {
            let sw = row_loss_w.map_or(1.0, |w| w[row]).sqrt();
            let whiten_row = |r: ArrayView1<'_, f64>| -> Vec<f64> {
                match metric {
                    Some(m) if whitens => {
                        let mut w = m.whiten_residual_row(row, r);
                        for x in w.iter_mut() {
                            *x *= sw;
                        }
                        w
                    }
                    _ => r.iter().map(|&x| x * sw).collect(),
                }
            };
            let wtarget = whiten_row(target.row(row));
            for kk in 0..k {
                wdesign[kk] = whiten_row(designs[kk].row(row));
            }
            for j in 0..k {
                rhs[j] += wtarget
                    .iter()
                    .zip(wdesign[j].iter())
                    .map(|(t, d)| t * d)
                    .sum::<f64>();
                for kk in j..k {
                    let g: f64 = wdesign[j]
                        .iter()
                        .zip(wdesign[kk].iter())
                        .map(|(x, y)| x * y)
                        .sum();
                    gram[[j, kk]] += g;
                    if kk != j {
                        gram[[kk, j]] += g;
                    }
                }
            }
        }
        // Numerical PD epsilon (NOT a penalty): keep the Cholesky well-posed when
        // two atoms' contributions are near-collinear (co-collapse). The
        // non-negativity clamp below — not this epsilon — is what bounds the
        // amplitude on a rank-deficient design, so the epsilon can stay at
        // machine scale and never biases a well-identified amplitude.
        let scale = (0..k)
            .map(|j| gram[[j, j]])
            .fold(0.0_f64, f64::max)
            .max(1e-300);
        for j in 0..k {
            gram[[j, j]] += 1e-12 * scale;
        }
        let raw = gram
            .cholesky(Side::Lower)
            .map_err(|e| format!("optimize_log_amplitudes_closed_form: cholesky failed: {e}"))?
            .solvevec(&rhs);
        // Project to the non-negative orthant (guardrail: born atom takes only its
        // residual-appropriate share; a negative optimum means the atom is off).
        // For an already-non-negative optimum this is exact; where it clamps, the
        // remaining atoms' share is re-absorbed on the next accepted-iterate solve.
        const AMP_FLOOR: f64 = 1.0e-12;
        for atom_idx in 0..k {
            let b = raw[atom_idx];
            self.atoms[atom_idx].log_amplitude = if b.is_finite() && b > AMP_FLOOR {
                b.ln()
            } else {
                AMP_FLOOR.ln()
            };
        }
        let df_after = self
            .try_fitted_for_rho(rho)
            .and_then(|recon| self.data_fit_for_reconstruction(target, recon.view()))
            .unwrap_or(f64::INFINITY);
        // Keep-best on the fit's own weighted data-fit (lower = better). Tolerance
        // is relative to the incumbent so a benign round-off wobble is accepted but
        // a genuine degradation reverts. `df_before` finite is guaranteed unless the
        // entry reconstruction itself failed, in which case any finite `df_after`
        // is an improvement over the +inf sentinel and is kept.
        let tol = 1.0e-9 * df_before.abs().max(1.0);
        if !(df_after <= df_before + tol) {
            for (j, atom) in self.atoms.iter_mut().enumerate() {
                atom.log_amplitude = saved[j];
            }
        }
        Ok(())
    }
}

// ===========================================================================
// F1 — amplitude-concentration certificate (the "intensity is presence vs a
// hidden radial coordinate" law).
//
// The scale/intensity/existence quotient above splits an atom's magnitude into
// the unit-Frobenius decoder shape, the explicit log-amplitude `s_k`, and the
// gate. What it does NOT certify is the SHAPE of an atom's realized amplitude
// distribution ACROSS the samples it fires on. Two regimes are observationally
// distinct and carry opposite structural verdicts:
//
//   * **Spike-at-saturation** — the realized amplitude piles at the two ends of
//     its range (near 0 = absent, near its saturation = present). This is a
//     genuine binary presence coordinate; the gate is honest and the atom's
//     latent dimension is what the chart says it is (a `circle` stays a circle).
//   * **Continuous** — the amplitude spreads unimodally across the interior of
//     its range. Intensity is then not presence but a hidden RADIAL latent axis:
//     the atom is really a disk / annulus (`S¹ × ℝ_radius`), and `d_atom` is
//     understated by one. `steer_delta`'s predicted nats scale with `a²`, so a
//     dosimetry claim rides on this uncertified quantity unless the radial axis
//     is promoted to an explicit coordinate and raced (circle vs cylinder-radial
//     vs disk).
//
// The certificate is an EVIDENCE decision, not a tuned threshold. Normalise the
// realized amplitudes to their saturation `r = a / max(a) ∈ (0, 1)` and fit a
// Beta(α, β) by maximum likelihood. The Beta family's own analytic mode-count
// transition IS the decision boundary: `Beta(α, β)` is U-shaped (density → ∞ at
// BOTH endpoints, an interior minimum — mass at absent AND saturated) exactly
// when `α < 1 AND β < 1`, and is unimodal / monotone (mass in the interior — a
// radial spread) otherwise. The boundary `α = β = 1` is the uniform density, the
// analytic shape-transition of the family, so "spike vs continuous" is read off
// the fitted shape with no magic constant. A disk's area-uniform radius has
// density `∝ r = Beta(2, 1)` (α > 1 ⇒ Continuous), and a present/absent atom
// collapses onto both endpoints (α, β < 1 ⇒ SpikeAtSaturation) — both verdicts
// fall out of the family analytically.
// ===========================================================================

/// The certified verdict on one atom's realized amplitude-concentration law.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AmplitudeConcentration {
    /// The realized amplitude is bimodal at the ends of its range (present /
    /// absent): a genuine binary presence coordinate. The gate is honest and the
    /// atom keeps its charted latent dimension.
    SpikeAtSaturation,
    /// The realized amplitude spreads continuously across the interior: intensity
    /// is a hidden RADIAL latent axis. Promote radius to an explicit coordinate
    /// and race the atom as circle vs cylinder-radial vs disk.
    Continuous,
    /// Too few / degenerate (no spread, non-finite, or all-equal) amplitudes to
    /// certify. Carries no radial promotion — a constant-intensity atom is a pure
    /// presence coordinate, not a disk.
    Indeterminate,
}

impl AmplitudeConcentration {
    /// Lowercase label for the diagnostics payload.
    pub fn label(self) -> &'static str {
        match self {
            AmplitudeConcentration::SpikeAtSaturation => "spike_at_saturation",
            AmplitudeConcentration::Continuous => "continuous",
            AmplitudeConcentration::Indeterminate => "indeterminate",
        }
    }
}

/// The per-atom amplitude-concentration certificate (F1): the fitted Beta shape
/// of the realized amplitude distribution and the presence-vs-radial verdict it
/// implies. Produced by [`amplitude_concentration_certificate`].
#[derive(Debug, Clone, Copy)]
pub struct AmplitudeConcentrationCertificate {
    /// The certified verdict.
    pub verdict: AmplitudeConcentration,
    /// Fitted Beta shape parameter `α` of the saturation-normalized amplitudes.
    /// `NaN` when [`AmplitudeConcentration::Indeterminate`].
    pub beta_alpha: f64,
    /// Fitted Beta shape parameter `β`.
    pub beta_beta: f64,
    /// The Beta log-likelihood at `(α, β)` — the evidence the verdict is read
    /// from. `NaN` when indeterminate.
    pub log_likelihood: f64,
    /// Number of realized amplitudes the certificate was fitted from.
    pub n: usize,
}

impl AmplitudeConcentrationCertificate {
    /// `true` iff the certificate calls for promoting a radial latent axis: the
    /// amplitude is a continuous (radial) coordinate, not a binary presence.
    pub fn recommends_radial_axis(&self) -> bool {
        matches!(self.verdict, AmplitudeConcentration::Continuous)
    }
}

/// Certify one atom's realized amplitude-concentration law from the amplitudes
/// `a_n ≥ 0` it fires with across its samples (the gated intensity per row, e.g.
/// `exp(s_k)` times the per-row gate). The verdict is read from the fitted Beta
/// shape of the saturation-normalized amplitudes: U-shaped (`α < 1 ∧ β < 1`) ⟺
/// [`AmplitudeConcentration::SpikeAtSaturation`], otherwise
/// [`AmplitudeConcentration::Continuous`]; a degenerate / no-spread sample is
/// [`AmplitudeConcentration::Indeterminate`].
pub fn amplitude_concentration_certificate(
    amplitudes: ArrayView1<'_, f64>,
) -> AmplitudeConcentrationCertificate {
    let n = amplitudes.len();
    let indeterminate = |n: usize| AmplitudeConcentrationCertificate {
        verdict: AmplitudeConcentration::Indeterminate,
        beta_alpha: f64::NAN,
        beta_beta: f64::NAN,
        log_likelihood: f64::NAN,
        n,
    };
    if n < 4 {
        // Fewer than four samples cannot resolve a shape (a Beta has two shape
        // parameters; a bimodality claim needs mass observed at both ends).
        return indeterminate(n);
    }
    if amplitudes.iter().any(|a| !a.is_finite() || *a < 0.0) {
        return indeterminate(n);
    }
    let amax = amplitudes.iter().copied().fold(0.0_f64, f64::max);
    if !(amax > 0.0) {
        // All-zero: the atom never fires — no distribution to certify.
        return indeterminate(n);
    }
    // Saturation-normalize into [0, 1]. A near-constant amplitude (no spread)
    // carries neither bimodality nor a radial axis: it is a pure fixed-intensity
    // presence coordinate, reported Indeterminate so no radial axis is promoted.
    let raw: Vec<f64> = amplitudes.iter().map(|&a| (a / amax).clamp(0.0, 1.0)).collect();
    let mean_r: f64 = raw.iter().sum::<f64>() / n as f64;
    let var_r: f64 = raw.iter().map(|r| (r - mean_r).powi(2)).sum::<f64>() / n as f64;
    // Spread floor: the sample must vary by more than floating-point noise
    // relative to its scale for a shape to be identifiable at all.
    if !(var_r > f64::EPSILON) {
        return indeterminate(n);
    }
    // Open-interval boundary correction: map endpoints strictly inside (0, 1) via
    // the standard `(r(n−1) + 1/2)/n` compression so `ln r` / `ln(1−r)` stay
    // finite. This is a recognized boundary rule, not a tuning knob.
    let nf = n as f64;
    let r: Vec<f64> = raw
        .iter()
        .map(|&x| (x * (nf - 1.0) + 0.5) / nf)
        .collect();

    let (alpha, beta, loglik) = match fit_beta_mle(&r) {
        Some(v) => v,
        None => return indeterminate(n),
    };

    // The Beta family's analytic U-shape region: density diverges at both 0 and 1
    // (mass at absent AND saturation) iff both shape parameters are below the
    // uniform-density boundary `1`. This is the family's own mode-count
    // transition — the decision, not a threshold.
    let verdict = if alpha < 1.0 && beta < 1.0 {
        AmplitudeConcentration::SpikeAtSaturation
    } else {
        AmplitudeConcentration::Continuous
    };
    AmplitudeConcentrationCertificate {
        verdict,
        beta_alpha: alpha,
        beta_beta: beta,
        log_likelihood: loglik,
        n,
    }
}

/// Maximum-likelihood fit of a `Beta(α, β)` to samples `r ∈ (0, 1)` by Newton's
/// method on the (concave) Beta log-likelihood, method-of-moments initialized.
/// Returns `(α, β, loglik)` or `None` when the sufficient statistics are
/// undefined (a sample at the closed boundary slipped through, or the moments are
/// degenerate). Newton uses the exact digamma/trigamma score and Hessian — no
/// finite differences (SPEC), and no autodiff.
fn fit_beta_mle(r: &[f64]) -> Option<(f64, f64, f64)> {
    let n = r.len();
    if n < 2 {
        return None;
    }
    let mut sum_ln = 0.0_f64;
    let mut sum_ln1m = 0.0_f64;
    let mut mean = 0.0_f64;
    let mut mean_sq = 0.0_f64;
    for &x in r {
        if !(x > 0.0 && x < 1.0) {
            return None;
        }
        sum_ln += x.ln();
        sum_ln1m += (1.0 - x).ln();
        mean += x;
        mean_sq += x * x;
    }
    let nf = n as f64;
    mean /= nf;
    let var = (mean_sq / nf - mean * mean).max(f64::EPSILON);
    // Method-of-moments seed: `common = m(1−m)/v − 1`, `α = m·common`,
    // `β = (1−m)·common`. Guard positivity so Newton starts in the interior.
    let common = (mean * (1.0 - mean) / var - 1.0).max(1.0e-3);
    let mut alpha = (mean * common).max(1.0e-3);
    let mut beta = ((1.0 - mean) * common).max(1.0e-3);

    let s_ln = sum_ln / nf;
    let s_ln1m = sum_ln1m / nf;
    // Newton on the per-sample-averaged score (concave objective; the Hessian is
    // negative definite, so a damped Newton with step-halving converges).
    for _ in 0..100 {
        let psi_ab = digamma(alpha + beta);
        let g_a = s_ln - (digamma(alpha) - psi_ab);
        let g_b = s_ln1m - (digamma(beta) - psi_ab);
        if g_a.abs() < 1.0e-12 && g_b.abs() < 1.0e-12 {
            break;
        }
        let t_ab = trigamma(alpha + beta);
        // Negative Hessian of the averaged loglik (positive definite):
        //   H = [[ψ₁(α) − ψ₁(α+β), −ψ₁(α+β)], [−ψ₁(α+β), ψ₁(β) − ψ₁(α+β)]].
        let h_aa = trigamma(alpha) - t_ab;
        let h_bb = trigamma(beta) - t_ab;
        let h_ab = -t_ab;
        let det = h_aa * h_bb - h_ab * h_ab;
        if !(det.abs() > 0.0) {
            break;
        }
        // Newton step `Δ = H⁻¹ g` (H is the negative Hessian, g the gradient).
        let d_a = (h_bb * g_a - h_ab * g_b) / det;
        let d_b = (h_aa * g_b - h_ab * g_a) / det;
        // Step-halving to keep `(α, β)` strictly positive and non-decreasing in
        // loglik — a standard safeguard, no wall-clock budget.
        let mut step = 1.0_f64;
        let base = beta_loglik_avg(alpha, beta, s_ln, s_ln1m);
        let mut moved = false;
        for _ in 0..40 {
            let na = alpha + step * d_a;
            let nb = beta + step * d_b;
            if na > 0.0 && nb > 0.0 && beta_loglik_avg(na, nb, s_ln, s_ln1m) >= base {
                alpha = na;
                beta = nb;
                moved = true;
                break;
            }
            step *= 0.5;
        }
        if !moved {
            break;
        }
    }
    let loglik = nf * beta_loglik_avg(alpha, beta, s_ln, s_ln1m);
    if !loglik.is_finite() {
        return None;
    }
    Some((alpha, beta, loglik))
}

/// Per-sample-averaged Beta log-likelihood `(α−1)⟨ln r⟩ + (β−1)⟨ln(1−r)⟩ −
/// ln B(α, β)` given the averaged sufficient statistics.
fn beta_loglik_avg(alpha: f64, beta: f64, s_ln: f64, s_ln1m: f64) -> f64 {
    (alpha - 1.0) * s_ln + (beta - 1.0) * s_ln1m
        - (ln_gamma(alpha) + ln_gamma(beta) - ln_gamma(alpha + beta))
}

/// Digamma `ψ(x) = d/dx ln Γ(x)` for `x > 0`: recurrence up to `x ≥ 6` then the
/// standard asymptotic (Bernoulli) series. Hand-derived closed form.
fn digamma(mut x: f64) -> f64 {
    let mut result = 0.0_f64;
    // Recurse up to x ≥ 10 so the truncated Bernoulli tail is ~1e-11 (the x ≥ 6
    // cutoff leaves ~1e-6, too coarse for the Beta Newton and its own test).
    while x < 10.0 {
        result -= 1.0 / x;
        x += 1.0;
    }
    let inv = 1.0 / x;
    let inv2 = inv * inv;
    result + x.ln() - 0.5 * inv
        - inv2 * (1.0 / 12.0 - inv2 * (1.0 / 120.0 - inv2 / 252.0))
}

/// Trigamma `ψ₁(x) = d²/dx² ln Γ(x)` for `x > 0`: recurrence up to `x ≥ 6` then
/// the asymptotic series `1/x + 1/(2x²) + Σ B₂ₖ/x^{2k+1}`.
fn trigamma(mut x: f64) -> f64 {
    let mut result = 0.0_f64;
    // Same x ≥ 10 recurrence cutoff as `digamma` for ~1e-11 accuracy.
    while x < 10.0 {
        result += 1.0 / (x * x);
        x += 1.0;
    }
    let inv = 1.0 / x;
    let inv2 = inv * inv;
    result
        + inv * (1.0 + inv * (0.5 + inv * (1.0 / 6.0 - inv2 * (1.0 / 30.0 - inv2 / 42.0))))
}

/// `ln Γ(x)` for `x > 0` via the Lanczos approximation (g = 7). Hand-derived
/// closed form; used only to report the Beta log-likelihood.
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
    let mut a = C[0];
    let t = x + G - 0.5;
    for (i, &c) in C.iter().enumerate().skip(1) {
        a += c / (x + i as f64 - 1.0);
    }
    0.5 * (2.0 * std::f64::consts::PI).ln() + (x - 0.5) * t.ln() - t + a.ln()
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::{Array3, array};

    // ---- F1: amplitude-concentration certificate ----------------------------

    /// A deterministic low-discrepancy sequence on `[0, 1)` (van der Corput,
    /// base 2) so the amplitude tests need no RNG and are byte-reproducible.
    fn van_der_corput(n: usize) -> Vec<f64> {
        (0..n)
            .map(|i| {
                let (mut x, mut denom, mut k) = (0.0_f64, 2.0_f64, i + 1);
                while k > 0 {
                    x += (k & 1) as f64 / denom;
                    denom *= 2.0;
                    k >>= 1;
                }
                x
            })
            .collect()
    }

    #[test]
    fn digamma_trigamma_match_known_values() {
        // ψ(1) = −γ ≈ −0.5772156649; ψ(2) = 1 − γ; ψ₁(1) = π²/6.
        let gamma = 0.577_215_664_901_532_9_f64;
        assert!((digamma(1.0) + gamma).abs() < 1.0e-9);
        assert!((digamma(2.0) - (1.0 - gamma)).abs() < 1.0e-9);
        let pi2_6 = std::f64::consts::PI * std::f64::consts::PI / 6.0;
        assert!((trigamma(1.0) - pi2_6).abs() < 1.0e-8);
        // ln Γ(5) = ln 24.
        assert!((ln_gamma(5.0) - 24.0_f64.ln()).abs() < 1.0e-9);
    }

    #[test]
    fn beta_mle_recovers_planted_shape() {
        // Sample the Beta(2, 5) CDF quantiles deterministically via a coarse
        // inverse-CDF over a fine low-discrepancy grid on the density, and check
        // the MLE lands near the planted shape. We synthesize from the density
        // directly by rejection on the grid to stay RNG-free.
        // Simpler + exact: fit to the Beta(2,1) family whose CDF is r² so the
        // quantile of a uniform u is sqrt(u) — an exact inverse transform.
        let u = van_der_corput(400);
        let samples: Vec<f64> = u.iter().map(|&x| x.sqrt()).collect(); // Beta(2,1)
        let (a, b, _ll) = fit_beta_mle(&samples).expect("beta fit");
        assert!((a - 2.0).abs() < 0.3, "alpha {a}");
        assert!((b - 1.0).abs() < 0.3, "beta {b}");
    }

    #[test]
    fn continuous_disk_radius_recommends_radial_axis() {
        // A disk uniform in AREA has radius density ∝ r on [0, 1] = Beta(2, 1),
        // whose quantile of uniform u is sqrt(u). Amplitude = radius. The
        // certificate must read this as a continuous (radial) coordinate.
        let u = van_der_corput(500);
        let amps = Array1::from_iter(u.iter().map(|&x| x.sqrt()));
        let cert = amplitude_concentration_certificate(amps.view());
        assert_eq!(cert.verdict, AmplitudeConcentration::Continuous, "{cert:?}");
        assert!(cert.recommends_radial_axis());
        assert!(cert.beta_alpha > 1.0, "alpha {}", cert.beta_alpha);
    }

    #[test]
    fn true_presence_certifies_spike_at_saturation() {
        // A genuine binary presence atom: roughly half the samples absent
        // (amplitude ≈ 0) and half saturated (≈ 1), with a little jitter so the
        // sample is not literally two atoms. Mass at both ends ⇒ U-shaped Beta
        // (α, β < 1) ⇒ SpikeAtSaturation, and NO radial axis is promoted.
        let jitter = van_der_corput(600);
        let amps = Array1::from_iter(jitter.iter().enumerate().map(|(i, &j)| {
            let base = if i % 2 == 0 { 0.0 } else { 1.0 };
            // Pull each sample toward its end by ≤ 8% so the piles stay at the
            // endpoints without ever leaving [0, 1].
            (base + if base == 0.0 { 0.08 * j } else { -0.08 * j }).clamp(0.0, 1.0)
        }));
        let cert = amplitude_concentration_certificate(amps.view());
        assert_eq!(
            cert.verdict,
            AmplitudeConcentration::SpikeAtSaturation,
            "{cert:?}"
        );
        assert!(!cert.recommends_radial_axis());
        assert!(cert.beta_alpha < 1.0 && cert.beta_beta < 1.0, "{cert:?}");
    }

    #[test]
    fn degenerate_amplitudes_are_indeterminate() {
        // No spread (constant intensity) ⇒ pure fixed-intensity presence, not a
        // disk: Indeterminate, no radial promotion.
        let flat = Array1::from_elem(50, 0.7);
        let cert = amplitude_concentration_certificate(flat.view());
        assert_eq!(cert.verdict, AmplitudeConcentration::Indeterminate);
        assert!(!cert.recommends_radial_axis());
        // All-zero (never fires) is also indeterminate.
        let zero = Array1::<f64>::zeros(50);
        assert_eq!(
            amplitude_concentration_certificate(zero.view()).verdict,
            AmplitudeConcentration::Indeterminate
        );
        // Too few samples.
        let few = array![0.1, 0.9];
        assert_eq!(
            amplitude_concentration_certificate(few.view()).verdict,
            AmplitudeConcentration::Indeterminate
        );
    }

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
            coords: ArrayView2<'_, f64>,
        ) -> Option<Result<ndarray::Array4<f64>, String>> {
            if coords.ncols() != 1 {
                return Some(Err(format!(
                    "AffineLineEvaluator::second_jet_dyn: d = 1 evaluator got {} coords",
                    coords.ncols()
                )));
            }
            None
        }

        fn third_jet_dyn(
            &self,
            coords: ArrayView2<'_, f64>,
        ) -> Option<Result<ndarray::Array5<f64>, String>> {
            if coords.ncols() != 1 {
                return Some(Err(format!(
                    "AffineLineEvaluator::third_jet_dyn: d = 1 evaluator got {} coords",
                    coords.ncols()
                )));
            }
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
        assert!(
            worst < 1e-12,
            "radial gradient must project to 0, got {worst}"
        );

        let tangent = array![[0.0_f64, 1.0], [-1.0, 0.0]]; // ⟨tangent, B⟩ = 0
        let proj_t = unit_frobenius_tangent_projection(b.view(), tangent.view());
        let drift = proj_t
            .iter()
            .zip(tangent.iter())
            .map(|(a, c)| (a - c).abs())
            .fold(0.0_f64, f64::max);
        assert!(
            drift < 1e-12,
            "tangent gradient must pass through, drift {drift}"
        );
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
        assert!(
            (ed - (4.0_f64).sqrt()).abs() < 1e-9,
            "dense ratio must be √K"
        );
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
        let before = sample_decoded_curve(
            &ev,
            atom.decoder_coefficients.view(),
            atom.log_amplitude,
            coords.column(0),
        )
        .unwrap();
        let mut atom = atom;
        let applied = retract_decoder_unit_frobenius(&mut atom);
        assert!(applied, "a non-unit decoder must be retracted");
        let norm = decoder_frobenius_norm(atom.decoder_coefficients.view());
        assert!(
            (norm - 1.0).abs() < 1e-12,
            "‖B‖_F must be pinned to 1, got {norm}"
        );
        // Image after — exp(s)·Φ·B must be byte-close to the original.
        let after = sample_decoded_curve(
            &ev,
            atom.decoder_coefficients.view(),
            atom.log_amplitude,
            coords.column(0),
        )
        .unwrap();
        let drift = before
            .iter()
            .zip(after.iter())
            .map(|(a, b)| (a - b).abs())
            .fold(0.0_f64, f64::max);
        assert!(
            drift < 1e-10,
            "retraction must be image-frozen, drift {drift}"
        );
        // Idempotent: a second retraction is a no-op.
        assert!(
            !retract_decoder_unit_frobenius(&mut atom),
            "retraction must be idempotent"
        );
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
        // t_a = -t_b + 1  ⇒  physical point = (1 - t_b)·d. Matched grid so every
        // B point coincides with an A grid point (nearest-match is exact and the
        // reflected transition is recovered to machine precision).
        let cb = Array1::linspace(0.0, 1.0, 11);
        let db = array![[0.6_f64, 0.8], [-0.6, -0.8]]; // γ_b(t_b) = (0.6,0.8) + t_b·(-0.6,-0.8)
        let pts_b = sample_decoded_curve(&ev, db.view(), 0.0, cb.view()).unwrap();
        let tr = affine_chart_transition(pts_a.view(), ca.view(), pts_b.view(), cb.view(), None)
            .unwrap();
        assert!(
            (tr.slope + 1.0).abs() < 1e-6,
            "slope must be -1, got {}",
            tr.slope
        );
        assert!(
            (tr.offset - 1.0).abs() < 1e-6,
            "offset must be 1, got {}",
            tr.offset
        );
        assert!(
            tr.coord_residual < 1e-6,
            "coord residual {}",
            tr.coord_residual
        );
        assert!(
            tr.geometric_residual < 1e-6,
            "geometric residual {}",
            tr.geometric_residual
        );
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
        assert!(
            tr.geometric_residual > 1.0,
            "disjoint curve must have large geometric residual"
        );
        assert!(
            !tr.same_manifold(1.0, 1e-2),
            "disjoint curve must be rejected"
        );
    }
}
