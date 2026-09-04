use super::*;

/// The probability-integral transform of the fitted conditional transformation
/// model, returned on the standard-normal (score) scale.
///
/// gam#2600: the fitted density is the most-likely-transformation density
/// `φ(h) · h'`, so the model's own CDF is `F(y | x) = Φ(h(y, x))` — a proper CDF
/// on the whole real line. The PIT is therefore `u = Φ(h)` and the calibrated
/// score is `Φ⁻¹(u) = h`, clipped to the representable quantile window.
///
/// It used to be `u = (Φ(h) − Φ(l)) / (Φ(u) − Φ(l))`, the PIT of the law
/// CONDITIONED on `y ∈ [y_lo, y_hi]`. Two things came with that form and both go
/// away here: it saturates to exactly `0`/`1` off the fitted knot range (hence
/// the `OutsideCertifiedDomain` refusal that used to guard it — a legitimate
/// out-of-range prediction is now a legitimate saturated probability, not an
/// error), and — see `endpoint_normalizer` — it is what made the objective it
/// scores non-concave and non-coercive.
///
/// `clip_eps` bounds the reported score away from `±∞` for responses in the
/// extreme tails of the fitted transform.
pub fn transformation_normal_pit_score(h: f64, clip_eps: f64) -> Result<f64, String> {
    if !(clip_eps.is_finite() && clip_eps > 0.0 && clip_eps < 0.5) {
        return Err(TransformationNormalError::InvalidInput {
            reason: format!(
                "transformation-normal PIT requires clip_eps in (0, 0.5), got {clip_eps}"
            ),
        }
        .into());
    }
    if !h.is_finite() {
        return Err(TransformationNormalError::InvalidInput {
            reason: format!("transformation-normal PIT requires a finite h, got h={h}"),
        }
        .into());
    }
    // `Φ⁻¹(Φ(h)) = h` exactly in the model, so clipping the PIT PROBABILITY to
    // `[clip_eps, 1 − clip_eps]` is clipping the SCORE to the matching quantile
    // window. Applying it here avoids a `Φ` / `Φ⁻¹` round trip that would lose
    // digits in exactly the tails the clip exists to bound. Both ends are read
    // from the quantile kernel rather than one being negated: the kernel is not
    // bit-antisymmetric, and the emitted score has to be the value a caller gets
    // from `standard_normal_quantile(clip_eps)` when it computes the same
    // boundary itself (`score_influence_jacobian` does, to detect saturation).
    let lower = standard_normal_quantile(clip_eps)
        .map_err(|err| format!("transformation-normal PIT lower clip bound failed: {err}"))?;
    let upper = standard_normal_quantile(1.0 - clip_eps)
        .map_err(|err| format!("transformation-normal PIT upper clip bound failed: {err}"))?;
    Ok(h.clamp(lower, upper))
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
            assert!(transformation_normal_pit_score(0.0, bad).is_err());
        }
        // A value strictly inside (0, 0.5) is accepted.
        assert!(transformation_normal_pit_score(0.0, 0.25).is_ok());
    }

    #[test]
    fn pit_rejects_nonfinite_h() {
        assert!(transformation_normal_pit_score(f64::NAN, EPS).is_err());
        assert!(transformation_normal_pit_score(f64::NEG_INFINITY, EPS).is_err());
        assert!(transformation_normal_pit_score(f64::INFINITY, EPS).is_err());
    }

    // ---- transformation_normal_pit_score: closed-form values ----

    #[test]
    fn pit_score_is_the_transform_itself_inside_the_clip_window() {
        // gam#2600: `F = Φ(h)`, so `Φ⁻¹(F) = h` exactly. The score is the
        // transform, with no endpoint arithmetic anywhere in it.
        for h in [-2.5, -0.5, 0.0, 0.25, 1.75] {
            let score = transformation_normal_pit_score(h, EPS).unwrap();
            assert_eq!(score, h, "PIT score must be h itself, got {score} for {h}");
        }
    }

    #[test]
    fn pit_score_clips_to_the_quantile_window_in_both_tails() {
        let clip = 1e-6;
        let upper = standard_normal_quantile(1.0 - clip).unwrap();
        let lower = standard_normal_quantile(clip).unwrap();
        assert!(upper > 4.0 && lower < -4.0);
        assert_eq!(
            transformation_normal_pit_score(1.0e3, clip).unwrap(),
            upper,
            "an extreme upper-tail response clips to Φ⁻¹(1 − clip_eps)"
        );
        assert_eq!(
            transformation_normal_pit_score(-1.0e3, clip).unwrap(),
            lower,
            "and the lower tail clips to Φ⁻¹(clip_eps)"
        );
        // Both bounds come from the quantile kernel itself. It is not
        // bit-antisymmetric, so negating one end would put the emitted score a
        // few ULP away from the boundary a caller recomputes for itself — which
        // is how `score_influence_jacobian` decides a row is saturated.
        assert!(
            (lower + upper).abs() > 0.0,
            "if the kernel ever becomes bit-antisymmetric this note is stale, not wrong"
        );
    }

    #[test]
    fn pit_score_no_longer_refuses_an_out_of_range_response() {
        // Before gam#2600 an `h` past the fitted support endpoints was a typed
        // `OutsideCertifiedDomain` refusal, because the CONDITIONAL PIT
        // saturates to exactly 0/1 there and a clamped answer would have been
        // fabricated. `F = Φ(h)` has no such endpoint, so an out-of-range
        // response is an ordinary (if extreme) probability.
        let clip = 1e-9;
        for h in [-12.0, -6.0, 6.0, 12.0] {
            let score = transformation_normal_pit_score(h, clip)
                .expect("an out-of-range response is a probability, not an error");
            assert!(score.is_finite());
        }
    }

    #[test]
    fn pit_is_monotone_increasing_in_h() {
        let clip = 1e-9;
        let a = transformation_normal_pit_score(-0.5, clip).unwrap();
        let b = transformation_normal_pit_score(0.0, clip).unwrap();
        let c = transformation_normal_pit_score(0.5, clip).unwrap();
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
}

