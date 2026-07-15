use super::*;

/// Boundary-roundoff floor (in ULP of the endpoint magnitude) for the
/// certified-support gate in [`transformation_normal_pit_score`]. An honest
/// training-boundary row reconstructs `lower`/`upper` through a `p_resp`-term
/// basis-weighted α sum, accumulating `~p_resp·ε·scale` cancellation; this
/// budget sits comfortably above realistic `p_resp` (≈ tens of shape columns)
/// so genuine roundoff is snapped to the endpoint while any larger excursion
/// is refused as out-of-certified-domain extrapolation.
const PIT_CERTIFIED_DOMAIN_ROUNDOFF_ULPS: f64 = 256.0;

pub fn transformation_normal_pit_score(
    h: f64,
    lower: f64,
    upper: f64,
    clip_eps: f64,
) -> Result<f64, String> {
    if !(clip_eps.is_finite() && clip_eps > 0.0 && clip_eps < 0.5) {
        return Err(TransformationNormalError::InvalidInput {
            reason: format!(
                "transformation-normal PIT requires clip_eps in (0, 0.5), got {clip_eps}"
            ),
        }
        .into());
    }
    if !(h.is_finite() && lower.is_finite() && upper.is_finite()) {
        return Err(TransformationNormalError::InvalidInput { reason: format!(
            "transformation-normal PIT requires finite h/lower/upper, got h={h}, lower={lower}, upper={upper}"
        ) }.into());
    }
    if upper <= lower {
        return Err(TransformationNormalError::MonotonicityViolated { reason: format!(
            "transformation-normal PIT endpoint order violated: lower={lower:.6e}, upper={upper:.6e}"
        ) }.into());
    }

    // Certified-domain gate (direct-α cutover, gam#2306). Positivity /
    // monotonicity of the transform is certified only on the fitted rows —
    // the factored Khatri-Rao cone guarantees `h(y_i, x_i) ∈ [lower_i,
    // upper_i]` there — plus whatever domain certificate the persisted model
    // carries. A prediction whose transformed value `h` sits meaningfully
    // outside `[lower, upper]` is therefore an extrapolation past the
    // certified support (a test response beyond the training range, or a
    // covariate `x` where `α_k(x)` left the positivity cone). Fabricating an
    // extreme-tail quantile by clamping would ship a silent, uncertified
    // answer, so refuse it (typed) and name the offending value + domain.
    //
    // The only tolerated excursion is boundary roundoff: an honest
    // training-boundary row reconstructs `lower`/`upper` through a
    // `p_resp`-term basis-weighted α sum, accumulating ~`p_resp·ε·scale`
    // cancellation. Snap that noise to the exact endpoint (it drives the
    // legitimate `u → {0, 1}` saturation the fit-time score paths consume),
    // and reject anything past the floor. Non-finite `h` is already rejected
    // above at the `is_finite()` guard.
    let support = upper - lower;
    let domain_scale = support.max(lower.abs()).max(upper.abs());
    let domain_tol = PIT_CERTIFIED_DOMAIN_ROUNDOFF_ULPS * f64::EPSILON * domain_scale;
    if h < lower - domain_tol {
        return Err(TransformationNormalError::OutsideCertifiedDomain { reason: format!(
            "transformation-normal PIT: transformed response h={h:.6e} lies below the certified \
             support lower endpoint {lower:.6e} by {:.6e} (> boundary-roundoff floor \
             {domain_tol:.3e}); positivity is certified only on the training support \
             [{lower:.6e}, {upper:.6e}], so this response/covariate is outside the fitted domain",
            lower - h
        ) }.into());
    }
    if h > upper + domain_tol {
        return Err(TransformationNormalError::OutsideCertifiedDomain { reason: format!(
            "transformation-normal PIT: transformed response h={h:.6e} lies above the certified \
             support upper endpoint {upper:.6e} by {:.6e} (> boundary-roundoff floor \
             {domain_tol:.3e}); positivity is certified only on the training support \
             [{lower:.6e}, {upper:.6e}], so this response/covariate is outside the fitted domain",
            h - upper
        ) }.into());
    }
    let h_inside = h.clamp(lower, upper);
    let u = if h_inside <= lower {
        0.0
    } else if h_inside >= upper {
        1.0
    } else {
        let log_num = log_normal_cdf_diff(h_inside, lower)?;
        let log_den = log_normal_cdf_diff(upper, lower)?;
        let ratio = (log_num - log_den).exp();
        if !(ratio.is_finite() && (-1.0e-12..=1.0 + 1.0e-12).contains(&ratio)) {
            return Err(TransformationNormalError::NumericalFailure { reason: format!(
                "transformation-normal PIT probability is not representable: h={h:.6e}, lower={lower:.6e}, upper={upper:.6e}, ratio={ratio}"
            ) }.into());
        }
        ratio.clamp(0.0, 1.0)
    };
    standard_normal_quantile(u.clamp(clip_eps, 1.0 - clip_eps))
        .map_err(|err| format!("transformation-normal PIT quantile failed: {err}"))
}

/// Accumulates the second-order monotone-transform quantities
/// `(h_i, h_j, h_ij, hp_i, hp_j, hp_ij)` for one row from the response value /
/// derivative bases and the per-response-knot ψ-directional derivatives of the
/// factored coordinates `α_k(x; ψ)`. With the direct-α chart (gam#2306) the
/// transform is LINEAR in the coordinates, so each accumulation is a plain
/// basis-weighted sum; the ψψ second derivative flows entirely through
/// `alpha_ij` (the covariate design is still nonlinear in ψ).
/// Shared verbatim across the SCOP Hessian/HVP/bilinear row loops.
pub(crate) fn scop_second_order_h(
    rv: ArrayView1<'_, f64>,
    rd: ArrayView1<'_, f64>,
    p_resp: usize,
    alpha_i: &[f64],
    alpha_j: &[f64],
    alpha_ij: &[f64],
) -> [f64; 6] {
    let mut h_i = 0.0;
    let mut h_j = 0.0;
    let mut h_ij = 0.0;
    let mut hp_i = 0.0;
    let mut hp_j = 0.0;
    let mut hp_ij = 0.0;
    for k in 0..p_resp {
        h_i += rv[k] * alpha_i[k];
        h_j += rv[k] * alpha_j[k];
        h_ij += rv[k] * alpha_ij[k];
        hp_i += rd[k] * alpha_i[k];
        hp_j += rd[k] * alpha_j[k];
        hp_ij += rd[k] * alpha_ij[k];
    }
    [h_i, h_j, h_ij, hp_i, hp_j, hp_ij]
}

/// Accumulates the second-order endpoint-normalizer chain inputs
/// `(endpoint_i, endpoint_j, endpoint_ij)` for one row. Shared verbatim across
/// the SCOP Hessian/HVP/bilinear row loops.
pub(crate) fn scop_second_order_endpoints(
    endpoint_basis: [&[f64]; 2],
    p_resp: usize,
    alpha_i: &[f64],
    alpha_j: &[f64],
    alpha_ij: &[f64],
) -> ([f64; 2], [f64; 2], [f64; 2]) {
    let mut endpoint_i = [0.0; 2];
    let mut endpoint_j = [0.0; 2];
    let mut endpoint_ij = [0.0; 2];
    for e in 0..2 {
        let basis = endpoint_basis[e];
        for k in 0..p_resp {
            endpoint_i[e] += basis[k] * alpha_i[k];
            endpoint_j[e] += basis[k] * alpha_j[k];
            endpoint_ij[e] += basis[k] * alpha_ij[k];
        }
    }
    (endpoint_i, endpoint_j, endpoint_ij)
}

/// Accumulates the psi-direction transform quantities `(h_psi, hp_psi,
/// endpoint_psi)` for one row from the response bases and the per-knot psi
/// directional derivatives of α. Shared verbatim across the SCOP psi setup
/// loops.
pub(crate) fn scop_psi_marginal(
    rv: ArrayView1<'_, f64>,
    rd: ArrayView1<'_, f64>,
    p_resp: usize,
    endpoint_basis: [&[f64]; 2],
    alpha_psi: &[f64],
) -> (f64, f64, [f64; 2]) {
    let mut h_psi = 0.0;
    let mut hp_psi = 0.0;
    for k in 0..p_resp {
        h_psi += rv[k] * alpha_psi[k];
        hp_psi += rd[k] * alpha_psi[k];
    }

    let mut endpoint_psi = [0.0; 2];
    for e in 0..2 {
        let basis = endpoint_basis[e];
        for k in 0..p_resp {
            endpoint_psi[e] += basis[k] * alpha_psi[k];
        }
    }
    (h_psi, hp_psi, endpoint_psi)
}

// ---------------------------------------------------------------------------
// Construction
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::array;

    const EPS: f64 = 1.0e-12;

    // ---- transformation_normal_pit_score: error semantics ----

    #[test]
    fn pit_rejects_clip_eps_outside_open_half_interval() {
        // clip_eps must satisfy 0 < clip_eps < 0.5.
        for bad in [0.0, -1.0e-3, 0.5, 0.6, f64::NAN, f64::INFINITY] {
            assert!(transformation_normal_pit_score(0.0, -1.0, 1.0, bad).is_err());
        }
        // A value strictly inside (0, 0.5) is accepted.
        assert!(transformation_normal_pit_score(0.0, -1.0, 1.0, 0.25).is_ok());
    }

    #[test]
    fn pit_rejects_nonfinite_h_lower_upper() {
        assert!(transformation_normal_pit_score(f64::NAN, -1.0, 1.0, EPS).is_err());
        assert!(transformation_normal_pit_score(0.0, f64::NEG_INFINITY, 1.0, EPS).is_err());
        assert!(transformation_normal_pit_score(0.0, -1.0, f64::INFINITY, EPS).is_err());
    }

    #[test]
    fn pit_rejects_endpoint_order_violation() {
        // upper <= lower is a monotonicity violation.
        assert!(transformation_normal_pit_score(0.0, 1.0, 1.0, EPS).is_err());
        assert!(transformation_normal_pit_score(0.0, 1.0, 0.5, EPS).is_err());
    }

    // ---- transformation_normal_pit_score: closed-form values ----

    #[test]
    fn pit_symmetric_midpoint_maps_to_zero() {
        // lower = -upper and h = 0: by normal symmetry the conditional CDF is
        // exactly 0.5, and Phi^{-1}(0.5) = 0.
        let u = transformation_normal_pit_score(0.0, -2.0, 2.0, EPS).unwrap();
        assert!(u.abs() < 1e-9, "expected ~0, got {u}");
    }

    #[test]
    fn pit_at_lower_endpoint_saturates_but_below_domain_refuses() {
        // In-domain path unchanged: h exactly at (or within the roundoff floor
        // of) `lower` saturates u -> 0 -> clip_eps -> Phi^{-1}(clip_eps). This
        // is the legitimate boundary saturation the fit-time score paths
        // consume, so it must keep working.
        let clip = 1e-6;
        let expected = standard_normal_quantile(clip).unwrap();
        let at = transformation_normal_pit_score(-1.0, -1.0, 1.0, clip).unwrap();
        assert!((at - expected).abs() < 1e-12);
        assert!(expected < -3.0);
        // A tiny sub-floor roundoff excursion still snaps to the endpoint.
        let roundoff = -1.0 - 8.0 * f64::EPSILON;
        let at_roundoff = transformation_normal_pit_score(roundoff, -1.0, 1.0, clip).unwrap();
        assert!((at_roundoff - expected).abs() < 1e-12);

        // Direct-α cutover (gam#2306): a genuine out-of-certified-domain probe
        // below `lower` is a typed refusal naming the value and the domain, not
        // a clamped tail quantile.
        let err = transformation_normal_pit_score(-1.5, -1.0, 1.0, clip)
            .expect_err("h far below lower must refuse, not clamp");
        assert!(err.contains("certified"), "message names the domain: {err}");
        assert!(err.contains("-1.500000e0"), "message names h: {err}");
        assert!(err.contains("outside the fitted domain"), "message: {err}");
    }

    #[test]
    fn pit_at_upper_endpoint_saturates_but_above_domain_refuses() {
        // In-domain path unchanged: h at (or within the roundoff floor of)
        // `upper` saturates u -> 1 -> 1 - clip_eps -> Phi^{-1}(1 - clip_eps).
        let clip = 1e-6;
        let expected = standard_normal_quantile(1.0 - clip).unwrap();
        let at = transformation_normal_pit_score(1.0, -1.0, 1.0, clip).unwrap();
        assert!((at - expected).abs() < 1e-12);
        assert!(expected > 3.0);
        let roundoff = 1.0 + 8.0 * f64::EPSILON;
        let at_roundoff = transformation_normal_pit_score(roundoff, -1.0, 1.0, clip).unwrap();
        assert!((at_roundoff - expected).abs() < 1e-12);

        // A genuine out-of-certified-domain probe above `upper` refuses.
        let err = transformation_normal_pit_score(2.0, -1.0, 1.0, clip)
            .expect_err("h far above upper must refuse, not clamp");
        assert!(err.contains("certified"), "message names the domain: {err}");
        assert!(err.contains("2.000000e0"), "message names h: {err}");
        assert!(err.contains("outside the fitted domain"), "message: {err}");
    }

    #[test]
    fn pit_is_monotone_increasing_in_h() {
        // Strictly inside the support the PIT score increases with h.
        let clip = 1e-9;
        let a = transformation_normal_pit_score(-0.5, -2.0, 2.0, clip).unwrap();
        let b = transformation_normal_pit_score(0.0, -2.0, 2.0, clip).unwrap();
        let c = transformation_normal_pit_score(0.5, -2.0, 2.0, clip).unwrap();
        assert!(a < b && b < c, "not monotone: {a} {b} {c}");
    }

    // ---- scop_second_order_h: pure accumulator closed forms ----

    #[test]
    fn scop_second_order_h_is_linear_in_the_directional_coordinates() {
        // Direct-α chart: every output is a plain basis-weighted sum of the
        // matching α directional-derivative slots — no coordinate factor.
        let rv = array![3.0];
        let rd = array![5.0];
        let ai = [2.0];
        let aj = [7.0];
        let aij = [11.0];
        let out = scop_second_order_h(rv.view(), rd.view(), 1, &ai, &aj, &aij);
        assert_eq!(
            out,
            [
                3.0 * 2.0,
                3.0 * 7.0,
                3.0 * 11.0,
                5.0 * 2.0,
                5.0 * 7.0,
                5.0 * 11.0
            ]
        );
    }

    #[test]
    fn scop_second_order_h_p_resp_two_matches_hand_formula() {
        let rv = array![1.0, 4.0];
        let rd = array![1.0, 6.0];
        let ai = [1.0, 3.0];
        let aj = [1.0, 5.0];
        let aij = [1.0, 7.0];
        let out = scop_second_order_h(rv.view(), rd.view(), 2, &ai, &aj, &aij);
        // h_i = rv0*ai0 + rv1*ai1 = 1 + 12 = 13; hp_i = 1 + 18 = 19; etc.
        assert_eq!(
            out,
            [
                1.0 + 4.0 * 3.0,
                1.0 + 4.0 * 5.0,
                1.0 + 4.0 * 7.0,
                1.0 + 6.0 * 3.0,
                1.0 + 6.0 * 5.0,
                1.0 + 6.0 * 7.0
            ]
        );
    }

    // ---- scop_second_order_endpoints ----

    #[test]
    fn scop_second_order_endpoints_matches_hand_formula() {
        let lower = [1.0, 2.0];
        let upper = [3.0, 4.0];
        let ai = [1.0, 6.0];
        let aj = [1.0, 7.0];
        let aij = [1.0, 8.0];
        let (ei, ej, eij) = scop_second_order_endpoints([&lower, &upper], 2, &ai, &aj, &aij);
        assert_eq!(ei[0], 1.0 + 2.0 * 6.0);
        assert_eq!(ei[1], 3.0 + 4.0 * 6.0);
        assert_eq!(ej[0], 1.0 + 2.0 * 7.0);
        assert_eq!(ej[1], 3.0 + 4.0 * 7.0);
        assert_eq!(eij[0], 1.0 + 2.0 * 8.0);
        assert_eq!(eij[1], 3.0 + 4.0 * 8.0);
    }

    // ---- scop_psi_marginal ----

    #[test]
    fn scop_psi_marginal_matches_hand_formula() {
        let rv = array![1.0, 4.0];
        let rd = array![1.0, 6.0];
        let lower = [1.0, 2.0];
        let upper = [3.0, 4.0];
        let alpha_psi = [9.0, 10.0];
        let (h_psi, hp_psi, endpoint_psi) =
            scop_psi_marginal(rv.view(), rd.view(), 2, [&lower, &upper], &alpha_psi);
        assert_eq!(h_psi, 9.0 + 4.0 * 10.0);
        assert_eq!(hp_psi, 9.0 + 6.0 * 10.0);
        assert_eq!(endpoint_psi[0], 1.0 * 9.0 + 2.0 * 10.0);
        assert_eq!(endpoint_psi[1], 3.0 * 9.0 + 4.0 * 10.0);
    }
}
