/// The end-to-end curvature-as-an-estimand report for one `curv(...)` smooth:
/// the fitted κ̂, its profile-likelihood confidence interval, the interior
/// κ = 0 likelihood-ratio flatness test, and the topology-free geometry
/// verdict. This is the #944 headline — it turns "we chose hyperbolic space"
/// into "κ̂ = −1.8 (95% CI −2.6, −1.1), flat rejected at p = …".
#[derive(Clone, Debug)]
pub struct CurvatureInference {
    /// Smooth-term index of the `curv(...)` term this report is about.
    pub term_idx: usize,
    /// The fitted signed sectional curvature κ̂ (the bounded analytic
    /// curvature profile optimum).
    pub kappa_hat: f64,
    /// Profile-likelihood CI for κ and the geometry verdict from its sign.
    pub ci: gam_geometry::curvature_estimand::KappaProfileCi,
    /// Interior-point κ = 0 likelihood-ratio flatness test (full χ²₁, no
    /// half-χ² boundary correction — κ = 0 is an interior point of the
    /// `S^d ← ℝ^d → H^d` family).
    pub flatness: gam_geometry::curvature_estimand::FlatnessTest,
    /// The kernel range `ℓ̂` the criterion profiles to AT `κ̂` — the smooth's
    /// second outer coordinate (gam#2747).
    ///
    /// It is reported rather than hidden because every statistic above is a
    /// PROFILE over it: `κ̂` is the argmin of `V_p(κ) = min_η V(κ, η)`, the CI
    /// is a profile-likelihood interval, and the flatness LR compares two
    /// range-profiled values. A reader who cannot see `ℓ̂` cannot tell an
    /// estimate anchored at a sensible resolution from one anchored at a
    /// degenerate corner of the range window.
    pub length_scale_hat: f64,
    /// Was `ℓ̂` estimated, or pinned by an explicit `length_scale=`?
    pub length_scale_estimated: bool,
    /// WHERE `ℓ̂` sits in the range chart (gam#2747). `length_scale_hat` alone
    /// cannot distinguish an interior minimum from an arrival at the
    /// geodesic-distance face from a stop at the evaluability wall, and the
    /// three support different claims about the magnitude — see
    /// [`gam_geometry::curvature_estimand::RangeEstimateSupport`]. This is the
    /// range's version of `ci.kappa_hat_support`, and it exists for the same
    /// reason: a provenance a reader has to infer is one they will get wrong.
    pub length_scale_support: gam_geometry::curvature_estimand::RangeEstimateSupport,
}

/// Compute the #944 curvature inference for the constant-curvature smooth at
/// `term_idx`, given the already-fitted resolved spec (carrying κ̂) and the same
/// fit inputs used to produce it.
///
/// The point estimate and inference share the same continuously smoothing-
/// profiled Gaussian REML evidence and its analytic profile jet. Each CI
/// endpoint solves the Wilks likelihood-ratio equation directly inside the
/// chart-bound bracket with safeguarded Newton steps; bisection is the
/// guaranteed-progress fallback.
///
/// Every decision here is made in the criterion's own units against its
/// statistical resolution `τ` ([`gam_solve::rho_optimizer::criterion_statistical_resolution`],
/// `1/(2n)`, #3245): an endpoint is resolved once the likelihood-ratio residual
/// at one end of the bracket is within `τ` of the threshold, and a descent is
/// refused only when the exact profile jet predicts a decrease larger than `τ`.
/// There is no separate relative or absolute tolerance to choose.
fn curvature_profile_lr_endpoint<F>(
    profile: &mut F,
    kappa_hat: f64,
    value_hat: f64,
    bound: f64,
    half_threshold: f64,
    resolution: f64,
) -> Result<(f64, bool), String>
where
    F: FnMut(f64) -> Result<(f64, f64, f64), String>,
{
    let direction = (bound - kappa_hat).signum();
    if direction == 0.0 {
        return Ok((bound, true));
    }

    // Nothing lies beyond the chart bound, so the profile's slope there makes
    // no claim about the likelihood set; only its value does.
    let (bound_value, bound_score, _) = profile(bound)?;
    if bound_value < value_hat - resolution {
        return Err(format!(
            "fitted curvature is not the minimum of its inference profile: \
             V(bound={bound})={bound_value:.6e} < V(kappa_hat)={value_hat:.6e} \
             by more than the criterion resolution {resolution:.6e}"
        ));
    }
    let bound_residual = bound_value - value_hat - half_threshold;
    if bound_residual < 0.0 {
        return Ok((bound, true));
    }
    if bound_residual == 0.0 {
        return Ok((bound, false));
    }

    // `inside` is in the connected likelihood set and `outside` is beyond its
    // first threshold crossing. Newton uses the exact profile score. It is
    // accepted only in the central half of the current bracket, so every other
    // iteration is a bisection-quality contraction even on a nearly flat score.
    // The bracket is resolved once either end's residual is within the
    // criterion resolution of the threshold: a value that close to the crossing
    // is not statistically distinguishable from it.
    let mut inside_x = kappa_hat;
    let mut inside_residual = -half_threshold;
    let mut outside_x = bound;
    let mut outside_residual = bound_residual;
    let mut outside_score = bound_score;
    while outside_residual > resolution && inside_residual < -resolution {
        let lo = inside_x.min(outside_x);
        let hi = inside_x.max(outside_x);
        let width = hi - lo;
        let central_lo = lo + 0.25 * width;
        let central_hi = hi - 0.25 * width;
        let newton = outside_x - outside_residual / outside_score;
        let probe = if newton.is_finite() && newton > central_lo && newton < central_hi {
            newton
        } else {
            lo + 0.5 * width
        };
        if !(probe > lo && probe < hi) {
            break;
        }
        let (value, score, curvature) = profile(probe)?;
        let residual = value - value_hat - half_threshold;
        if residual >= 0.0 {
            // Outside the likelihood set at any slope: the first crossing is in
            // (inside_x, probe], and what the profile does past the probe has
            // no bearing on it (gam#3509). On a multimodal profile it falls
            // toward another basin there, which the jet's model over the rest
            // of the bracket reads as a decrease (#1464).
            outside_x = probe;
            outside_residual = residual;
            outside_score = score;
        } else {
            // The search assumes the profile rises outward through the
            // likelihood set. A downward outward slope at a point inside it is
            // refused when the exact jet predicts a decrease larger than the
            // resolution over the part of the bracket still outward of the
            // point: that is an interior maximum past which the walk could
            // report a later crossing, a different component.
            let outward_score = direction * score;
            let outward_decrease =
                profile_model_decrease(outward_score, curvature, 0.0, (outside_x - probe).abs());
            if outward_decrease > resolution {
                return Err(format!(
                    "curvature profile changed direction inside its likelihood-ratio set at \
                     kappa={probe}: outward score {outward_score:.6e} and curvature \
                     {curvature:.6e} predict a decrease of {outward_decrease:.6e}, above the \
                     criterion resolution {resolution:.6e} (V(kappa)={value:.9e}, \
                     V(kappa) - V(kappa_hat) - half_threshold = {residual:.6e})"
                ));
            }
            inside_x = probe;
            inside_residual = residual;
        }
    }
    // `outside_x` already holds the analytic score and residual evaluated
    // there, so one final Newton step costs no additional profile evaluation
    // and resolves the crossing to the accuracy of the score itself. It is
    // taken only when it lands inside the certified bracket; otherwise the
    // midpoint stands.
    let midpoint = inside_x + 0.5 * (outside_x - inside_x);
    let refined = outside_x - outside_residual / outside_score;
    let lo = inside_x.min(outside_x);
    let hi = inside_x.max(outside_x);
    let endpoint = if refined.is_finite() && refined >= lo && refined <= hi {
        refined
    } else {
        midpoint
    };
    Ok((endpoint, false))
}

/// The largest decrease the exact second-order model `m(t) = s·t + ½·c·t²` of
/// the profile predicts over the feasible displacements `t ∈ [lower, upper]`
/// (`lower ≤ 0 ≤ upper`).
///
/// `c` enters with its own sign. A quadratic attains its minimum over an
/// interval either at its stationary point, which exists and is a minimum only
/// for `c > 0`, or at an endpoint, so the minimum is the least of those three
/// candidates. `t = 0` is feasible and `m(0) = 0`, so the decrease is never
/// negative.
///
/// Dropping a negative `c` instead — judging a concave point by its slope alone
/// — is NOT the conservative reading: it deletes the `½·c·t²` term that makes
/// `m` fall away, so it reports a SMALLER decrease and certifies a local
/// MAXIMUM along κ as an optimum whenever the slope there is small (#3453).
///
/// At an interior minimum this is `s²/(2c)`, half the squared Newton decrement
/// the outer certificate bounds; at a rail with the slope pointing out of the
/// box the feasible side has zero length and the decrease is exactly zero.
fn profile_model_decrease(score: f64, curvature: f64, lower: f64, upper: f64) -> f64 {
    let model = |t: f64| score * t + 0.5 * curvature * t * t;
    let mut lowest = model(lower).min(model(upper));
    if curvature > 0.0 {
        let stationary = -score / curvature;
        if stationary > lower && stationary < upper {
            lowest = lowest.min(model(stationary));
        }
    }
    -lowest
}

/// `value_ceiling` is a κ-free upper bound on the profile over the whole chart,
/// `V_p(κ) ≤ value_ceiling` for every κ in `[kappa_min, kappa_max]` (for the
/// constant-curvature smooth, [`ConstantCurvatureProfile::value_ceiling`]; pass
/// `f64::INFINITY` when none is known). When it sits within the Wilks level of
/// `V_p(κ̂)`, every κ in the chart satisfies `r(κ) = V_p(κ) − V_p(κ̂) − ½z² ≤ 0`:
/// the likelihood-ratio set IS the chart, exactly, so the interval is the chart
/// with both ends open and no endpoint search is needed (gam#3509). This is the
/// Wilks set itself, so it is calibrated rather than conservative, and it is the
/// only thing that identifies the set when the profile has an interior maximum
/// between κ̂ and a chart bound that no probe visits — the bound's own value and
/// slope cannot see such a bump.
fn curvature_profile_ci_from_analytic_score<F>(
    profile: &mut F,
    kappa_hat: f64,
    kappa_min: f64,
    kappa_max: f64,
    level: f64,
    resolution: f64,
    value_ceiling: f64,
) -> Result<gam_geometry::curvature_estimand::KappaProfileCi, String>
where
    F: FnMut(f64) -> Result<(f64, f64, f64), String>,
{
    if !(kappa_min < kappa_max && kappa_hat >= kappa_min && kappa_hat <= kappa_max) {
        return Err("curvature profile requires kappa_hat inside valid chart bounds".to_string());
    }
    if !(level > 0.0 && level < 1.0) {
        return Err("curvature profile level must lie in (0, 1)".to_string());
    }
    if !(resolution.is_finite() && resolution > 0.0) {
        return Err(format!(
            "curvature profile resolution must be finite and positive, got {resolution}"
        ));
    }
    let z = gam_geometry::curvature_estimand::wald_half_width(1.0, level)
        .ok_or_else(|| "curvature profile threshold is not finite".to_string())?;
    let half_threshold = 0.5 * z * z;
    let (value_hat, score_hat, curvature_hat) = profile(kappa_hat)?;
    // The bounded solve projects onto the box, so a railed κ̂ sits EXACTLY on
    // its face; no proximity tolerance is needed to recognise it. The rail is
    // carried on the report (#2687) because κ̂ there is a box readout, not an
    // estimate.
    let at_lower = kappa_hat == kappa_min;
    let at_upper = kappa_hat == kappa_max;
    let kappa_hat_support = if at_lower {
        gam_geometry::curvature_estimand::KappaEstimateSupport::RailedAtLowerBound
    } else if at_upper {
        gam_geometry::curvature_estimand::KappaEstimateSupport::RailedAtUpperBound
    } else {
        gam_geometry::curvature_estimand::KappaEstimateSupport::Interior
    };
    // Stationarity on the box: the decrease the exact profile jet predicts for
    // any feasible move away from κ̂ must be below the criterion resolution. At
    // a rail the infeasible side has zero length, so a score pointing out of
    // the box is stationary by construction — the rail relaxation is the
    // feasible set, not a separate rule.
    //
    // Except that the quadratic model, extrapolated across the whole box, is not
    // a statement about κ̂ (#1464). At a rail on a derived wall with the score
    // pointing out, every feasible move RAISES the profile to first order, and on
    // a concave profile the model first rises to `s²/(2|c|)` at `t = −s/c` before
    // it can fall. When that rise is itself resolvable, κ̂ is a box-KKT local
    // minimum that the model's far-side descent cannot reach without crossing it:
    // the far side is another basin, not a failure of this point's stationarity,
    // and whether it is LOWER is a question of value that the likelihood-ratio
    // walk below answers at the opposite bound (it refuses κ̂ if the bound is
    // below it). Measured: a five-centre κ*=+2 draw railed at the fold with
    // s = −12.8, c = −7.38 — a rise of 11.1 against a resolution of 8.3e-4 — and
    // the whole-box model's decrease of 13.7 refused it. A rise at or below the resolution is not
    // separated from a maximum along κ (#3453), so that case keeps the whole-box
    // model.
    let score_points_out = (at_upper && score_hat < 0.0) || (at_lower && score_hat > 0.0);
    let resolvable_rise = curvature_hat >= 0.0
        || score_hat * score_hat / (2.0 * curvature_hat.abs()) > resolution;
    let predicted_decrease = if score_points_out && resolvable_rise {
        0.0
    } else {
        profile_model_decrease(
            score_hat,
            curvature_hat,
            kappa_min - kappa_hat,
            kappa_max - kappa_hat,
        )
    };
    if predicted_decrease > resolution {
        // Name what was refused AGAINST, not just that something was refused.
        // A κ̂ that failed this check is either a genuine interior non-optimum or
        // a rail the solver did not land on, and the two need opposite repairs —
        // so the message has to carry the box and both gaps (#2687).
        return Err(format!(
            "curvature inference rejected a non-stationary point estimate: \
             kappa_hat={kappa_hat}, score={score_hat:.6e}, curvature={curvature_hat:.6e}, \
             predicted_decrease={predicted_decrease:.6e} > resolution={resolution:.6e}; \
             box=[{kappa_min}, {kappa_max}], gap_to_lower={:.6e}, gap_to_upper={:.6e}, \
             classified={}",
            kappa_hat - kappa_min,
            kappa_max - kappa_hat,
            kappa_hat_support.label()
        ));
    }

    if value_ceiling.is_nan() || value_hat > value_ceiling + resolution {
        // The ceiling is a theorem about the profile, so a κ̂ above it means the
        // ceiling's premise (the smooth switched off leaves only the intercept)
        // does not hold for this profile, not that κ̂ is fine.
        return Err(format!(
            "curvature profile exceeds its kappa-free ceiling: V(kappa_hat={kappa_hat})=\
             {value_hat:.9e}, ceiling={value_ceiling:.9e}, resolution={resolution:.6e}"
        ));
    }

    let ((ci_lo, lo_at_bound), (ci_hi, hi_at_bound)) =
        if value_ceiling - value_hat <= half_threshold {
            // sup_κ r(κ) ≤ value_ceiling − V_p(κ̂) − ½z² ≤ 0: the whole chart is
            // in the likelihood-ratio set, and both ends are the chart's walls.
            ((kappa_min, true), (kappa_max, true))
        } else {
            (
                curvature_profile_lr_endpoint(
                    profile,
                    kappa_hat,
                    value_hat,
                    kappa_min,
                    half_threshold,
                    resolution,
                )?,
                curvature_profile_lr_endpoint(
                    profile,
                    kappa_hat,
                    value_hat,
                    kappa_max,
                    half_threshold,
                    resolution,
                )?,
            )
        };
    let verdict = if ci_lo > 0.0 {
        gam_geometry::curvature_estimand::CurvatureVerdict::Spherical
    } else if ci_hi < 0.0 {
        gam_geometry::curvature_estimand::CurvatureVerdict::Hyperbolic
    } else {
        gam_geometry::curvature_estimand::CurvatureVerdict::Flat
    };
    Ok(gam_geometry::curvature_estimand::KappaProfileCi {
        kappa_hat,
        ci_lo,
        ci_hi,
        lo_at_bound,
        hi_at_bound,
        kappa_hat_support,
        verdict,
    })
}

pub fn curvature_inference_forspec(
    data: ArrayView2<'_, f64>,
    y: ArrayView1<'_, f64>,
    weights: ArrayView1<'_, f64>,
    offset: ArrayView1<'_, f64>,
    resolvedspec: &TermCollectionSpec,
    term_idx: usize,
    family: LikelihoodSpec,
    level: f64,
) -> Result<CurvatureInference, EstimationError> {
    let kappa_hat = get_constant_curvature_kappa(resolvedspec, term_idx).ok_or_else(|| {
        EstimationError::InvalidInput(format!(
            "curvature_inference_forspec: term {term_idx} is not a constant-curvature smooth"
        ))
    })?;
    if constant_curvature_kappa_is_fixed(resolvedspec, term_idx) {
        crate::bail_invalid_estim!(
            "curvature inference requires an estimated curvature; term {term_idx} has user-pinned kappa={kappa_hat}"
        );
    }
    if y.len() != data.nrows() || weights.len() != data.nrows() || offset.len() != data.nrows() {
        crate::bail_invalid_estim!(
            "curvature inference row mismatch: data={}, y={}, weights={}, offset={}",
            data.nrows(),
            y.len(),
            weights.len(),
            offset.len(),
        );
    }
    validate_constant_curvature_profile_inputs(resolvedspec, term_idx, weights, offset, &family)?;
    let (kappa_min, kappa_max) = constant_curvature_kappa_bounds(data, resolvedspec, term_idx);
    let (feature_cols, base_spec) = match resolvedspec
        .smooth_terms
        .get(term_idx)
        .map(|term| &term.basis)
    {
        Some(SmoothBasisSpec::ConstantCurvature {
            feature_cols, spec, ..
        }) => (feature_cols, spec.clone()),
        _ => {
            return Err(EstimationError::InvalidInput(format!(
                "constant-curvature κ profile: smooth term {term_idx} is not a \
                 constant-curvature basis"
            )));
        }
    };
    let x_term = select_columns(data, feature_cols).map_err(EstimationError::from)?;
    let profile = ConstantCurvatureProfile::new(x_term.view(), y, base_spec)?;
    let value_ceiling = profile.value_ceiling()?;

    // The profile is a total negative log-evidence, so its resolution is the
    // one the outer certificate that produced κ̂ used.
    let resolution = gam_solve::rho_optimizer::criterion_statistical_resolution(y.len())
        .ok_or_else(|| {
            EstimationError::InvalidInput("curvature inference requires observations".to_string())
        })?;
    // CI and flatness revisit κ̂ and κ=0. The shared profile caches each joint
    // value/analytic-jet triple so every statistic consumes the same evaluation.
    let mut v_p = |kappa: f64| -> Result<(f64, f64, f64), String> {
        if !kappa.is_finite() {
            return Err(format!("V_p probed a non-finite κ = {kappa}"));
        }
        profile.evaluate(kappa).map_err(|error| {
            format!("analytic curvature profile at kappa={kappa} failed: {error}")
        })
    };
    let ci = curvature_profile_ci_from_analytic_score(
        &mut v_p,
        kappa_hat,
        kappa_min,
        kappa_max,
        level,
        resolution,
        value_ceiling,
    )
    .map_err(EstimationError::RemlOptimizationFailed)?;
    let flatness = gam_geometry::curvature_estimand::flatness_lr_test(
        |kappa| v_p(kappa).map(|(value, _, _)| value),
        kappa_hat,
    )
    .map_err(EstimationError::RemlOptimizationFailed)?;

    let (eta_hat, _, range_outcome) = profile.minimize_over_eta(kappa_hat)?;
    Ok(CurvatureInference {
        term_idx,
        kappa_hat,
        ci,
        flatness,
        length_scale_hat: eta_hat.exp(),
        length_scale_estimated: profile.eta_bounds.is_some(),
        length_scale_support: range_outcome.support(),
    })
}

#[cfg(test)]
mod curvature_profile_score_tests {
    use super::*;

    /// The criterion resolution of a 1000-row fit.
    const TEST_RESOLUTION: f64 = 0.5 / 1000.0;

    #[test]
    fn analytic_profile_score_finds_exact_quadratic_lr_crossings() {
        let kappa_hat = -0.37;
        let curvature = 16.0;
        let level = 0.95;
        let resolution = TEST_RESOLUTION;
        let mut profile = |kappa: f64| -> Result<(f64, f64, f64), String> {
            let displacement = kappa - kappa_hat;
            Ok((
                7.0 + 0.5 * curvature * displacement * displacement,
                curvature * displacement,
                curvature,
            ))
        };
        let ci = curvature_profile_ci_from_analytic_score(
            &mut profile,
            kappa_hat,
            -3.0,
            3.0,
            level,
            resolution,
            f64::INFINITY,
        )
        .expect("analytic quadratic profile CI");
        let z = gam_geometry::curvature_estimand::wald_half_width(1.0, level)
            .expect("valid normal quantile");
        let expected_half_width = z / curvature.sqrt();
        // The bracket stops once the residual is within the resolution, i.e.
        // within `resolution / slope` of the crossing, with the slope there
        // `curvature · half_width`; the closing Newton step only tightens it.
        let endpoint_bound = resolution / (curvature * expected_half_width);
        assert!((ci.ci_lo - (kappa_hat - expected_half_width)).abs() <= endpoint_bound);
        assert!((ci.ci_hi - (kappa_hat + expected_half_width)).abs() <= endpoint_bound);
        assert!(!ci.lo_at_bound && !ci.hi_at_bound);
    }

    #[test]
    fn analytic_profile_marks_chart_bound_when_wilks_set_never_crosses() {
        let mut profile = |kappa: f64| -> Result<(f64, f64, f64), String> {
            Ok((0.5 * kappa * kappa, kappa, 1.0))
        };
        let ci = curvature_profile_ci_from_analytic_score(
            &mut profile,
            0.0,
            -0.1,
            0.1,
            0.95,
            TEST_RESOLUTION,
            f64::INFINITY,
        )
        .expect("open bounded profile CI");
        assert_eq!(ci.ci_lo, -0.1);
        assert_eq!(ci.ci_hi, 0.1);
        assert!(ci.lo_at_bound && ci.hi_at_bound);
        // κ̂ = 0 is interior to [−0.1, 0.1]: an open CI at both bounds is a
        // statement about the interval, not about the estimate.
        assert_eq!(
            ci.kappa_hat_support,
            gam_geometry::curvature_estimand::KappaEstimateSupport::Interior
        );
    }

    /// gam#2687: the analytic-score route already had to KNOW κ̂ was railed —
    /// its stationarity check relaxes to "the score points out of the box" at a
    /// bound, which is only sound for a boundary optimum — and then reported κ̂
    /// as an estimate anyway. Both halves are pinned here: the relaxed check
    /// still accepts, and the report now carries the rail.
    #[test]
    fn a_railed_point_estimate_is_accepted_and_declared_by_the_analytic_route_2687() {
        // V_p(κ) = −κ, score = −1: strictly decreasing, never stationary in the
        // interior. κ̂ can only be the upper bound.
        let kappa_max = 1.388_888_888_888_888_9_f64;
        let mut monotone =
            |kappa: f64| -> Result<(f64, f64, f64), String> { Ok((-kappa, -1.0, 0.0)) };
        let ci = curvature_profile_ci_from_analytic_score(
            &mut monotone,
            kappa_max,
            -kappa_max,
            kappa_max,
            0.95,
            TEST_RESOLUTION,
            f64::INFINITY,
        )
        .expect("a boundary optimum with the score pointing out of the box is stationary");
        assert_eq!(
            ci.kappa_hat_support,
            gam_geometry::curvature_estimand::KappaEstimateSupport::RailedAtUpperBound,
            "κ̂ = {kappa_max} is the box's own upper end"
        );
        // The mirrored sign, so the relaxation and the declaration agree on both
        // sides rather than one of them being written for a single branch.
        let mut increasing =
            |kappa: f64| -> Result<(f64, f64, f64), String> { Ok((kappa, 1.0, 0.0)) };
        let ci_lo = curvature_profile_ci_from_analytic_score(
            &mut increasing,
            -kappa_max,
            -kappa_max,
            kappa_max,
            0.95,
            TEST_RESOLUTION,
            f64::INFINITY,
        )
        .expect("the mirrored boundary optimum");
        assert_eq!(
            ci_lo.kappa_hat_support,
            gam_geometry::curvature_estimand::KappaEstimateSupport::RailedAtLowerBound
        );
        // An interior non-stationary point is still refused: the relaxation is
        // tied to the rail, not a blanket loosening.
        let mut interior_slope =
            |kappa: f64| -> Result<(f64, f64, f64), String> { Ok((-kappa, -1.0, 0.0)) };
        assert!(
            curvature_profile_ci_from_analytic_score(
                &mut interior_slope,
                0.0,
                -kappa_max,
                kappa_max,
                0.95,
                TEST_RESOLUTION,
                f64::INFINITY,
            )
            .is_err(),
            "a non-stationary INTERIOR point is not an optimum and must still be refused"
        );
    }

    /// gam#1464: at a rail on a derived wall, stationarity is local box-KKT. A
    /// concave profile railed at the upper wall with the score pointing out of the
    /// box and a resolvable inward rise is accepted, although the whole-box
    /// quadratic model predicts a descent at the far wall. The same shape with the
    /// score pointing INTO the box, and the same shape with a rise below the
    /// resolution, are still refused.
    #[test]
    fn a_rail_is_judged_by_local_box_kkt_not_the_whole_box_model_1464() {
        // `V = s·t − 2t² + t⁴` in the displacement `t = κ − 1` from the upper
        // wall. At the wall its jet is the concave quadratic `m(t) = s·t − 2t²`
        // (the quartic has no value, slope or curvature there). At s = −3 the
        // score points out of the box and `m` rises to 9/8 at t = −3/4 and falls
        // to −2 at the lower wall t = −2, so the whole-box model predicts a
        // descent. The profile itself does not descend: `V > 0` on all of
        // `[−2, 0)`, so the rail is its minimum over the box, as it is on a
        // profile the κ search has already compared across faces.
        let concave_at = |score: f64| {
            move |kappa: f64| -> Result<(f64, f64, f64), String> {
                let t = kappa - 1.0;
                Ok((
                    score * t - 2.0 * t * t + t.powi(4),
                    score - 4.0 * t + 4.0 * t.powi(3),
                    -4.0 + 12.0 * t * t,
                ))
            }
        };
        assert!(
            super::profile_model_decrease(-3.0, -4.0, -2.0, 0.0) > TEST_RESOLUTION,
            "the fixture must be one the whole-box model refuses"
        );
        let mut railed = concave_at(-3.0);
        let ci = curvature_profile_ci_from_analytic_score(
            &mut railed,
            1.0,
            -1.0,
            1.0,
            0.95,
            TEST_RESOLUTION,
            f64::INFINITY,
        )
        .expect("a rail with the score pointing out and a resolvable rise is a box-KKT minimum");
        assert_eq!(
            ci.kappa_hat_support,
            gam_geometry::curvature_estimand::KappaEstimateSupport::RailedAtUpperBound
        );

        let mut inward = concave_at(3.0);
        let error = curvature_profile_ci_from_analytic_score(
            &mut inward,
            1.0,
            -1.0,
            1.0,
            0.95,
            TEST_RESOLUTION,
            f64::INFINITY,
        )
        .expect_err("a rail whose score points into the box is not stationary");
        assert!(error.contains("non-stationary"), "{error}");

        // Out of the box, but the rise `s²/(2|c|)` is half the resolution: not
        // separated from a maximum along κ, so the whole-box model decides.
        let unresolved_score = -(TEST_RESOLUTION * 4.0).sqrt();
        assert!(unresolved_score * unresolved_score / 8.0 < TEST_RESOLUTION);
        let mut unresolved = concave_at(unresolved_score);
        let error = curvature_profile_ci_from_analytic_score(
            &mut unresolved,
            1.0,
            -1.0,
            1.0,
            0.95,
            TEST_RESOLUTION,
            f64::INFINITY,
        )
        .expect_err("an unresolvable rise leaves the whole-box model in charge");
        assert!(error.contains("non-stationary"), "{error}");

        // Off the rail the interior test decides: at the same profile's interior
        // point κ = 1/4 (t = −3/4) the score is −27/16 and the curvature 11/4,
        // a Newton decrease of about 0.52, far above the resolution, so it is
        // refused although no wall is involved.
        let mut interior = concave_at(-3.0);
        let error = curvature_profile_ci_from_analytic_score(
            &mut interior,
            0.25,
            -1.0,
            1.0,
            0.95,
            TEST_RESOLUTION,
            f64::INFINITY,
        )
        .expect_err("an interior maximum along κ is not an optimum");
        assert!(error.contains("non-stationary"), "{error}");
    }

    /// gam#3453: this layer used to re-test κ̂ against its own bar
    /// `|V_p'| ≤ max(rt, √ε)·(1 + |V_p|)`, which is in raw gradient units and
    /// scales with the ADDITIVE level of the criterion — so it refused κ̂ values
    /// the outer κ solve had already certified, and its verdict changed when a
    /// constant was added to `V_p`. Every check is now a predicted DECREASE in
    /// the criterion judged against the criterion's own resolution, which no
    /// additive constant moves. The reproducer's point is pinned at three levels
    /// spanning seven orders, together with the two shapes that must still be
    /// refused at every level.
    #[test]
    fn an_outer_certified_kappa_hat_is_accepted_at_the_decrement_resolution_3453() {
        let kappa_hat = -1.1759;
        let curvature = 59.0;
        let quadratic = |level: f64, gradient: f64| {
            move |kappa: f64| -> Result<(f64, f64, f64), String> {
                let d = kappa - kappa_hat;
                Ok((
                    level + gradient * d + 0.5 * curvature * d * d,
                    gradient + curvature * d,
                    curvature,
                ))
            }
        };
        // The issue's certified point: its predicted decrease `g²/(2H)` is five
        // orders below the resolution, so it is stationary at the bar the outer
        // solve certified it against.
        let certified_gradient = -4.04e-3;
        assert!(
            certified_gradient * certified_gradient / (2.0 * curvature) < TEST_RESOLUTION,
            "the reproducer's point must be stationary at the criterion resolution"
        );
        for level in [0.0, 5.8e2, 1.0e7] {
            let mut profile = quadratic(level, certified_gradient);
            let ci = curvature_profile_ci_from_analytic_score(
                &mut profile,
                kappa_hat,
                -3.0,
                3.0,
                0.95,
                TEST_RESOLUTION,
                f64::INFINITY,
            )
            .unwrap_or_else(|e| panic!("certified kappa_hat refused at V level {level}: {e}"));
            assert_eq!(
                ci.kappa_hat_support,
                gam_geometry::curvature_estimand::KappaEstimateSupport::Interior
            );
            assert!(ci.ci_lo < kappa_hat && kappa_hat < ci.ci_hi && ci.ci_hi < 0.0);
        }

        // A gradient whose predicted decrease exceeds the resolution is a real
        // missed descent, and is still refused at every level.
        let resolvable_gradient = 1.1 * (2.0 * TEST_RESOLUTION * curvature).sqrt();
        assert!(
            resolvable_gradient * resolvable_gradient / (2.0 * curvature) > TEST_RESOLUTION,
            "the refused arm must actually exceed the resolution"
        );
        for level in [0.0, 1.0e7] {
            let mut profile = quadratic(level, resolvable_gradient);
            let error = curvature_profile_ci_from_analytic_score(
                &mut profile,
                kappa_hat,
                -3.0,
                3.0,
                0.95,
                TEST_RESOLUTION,
                f64::INFINITY,
            )
            .expect_err("a resolvable decrease is not a stationary point");
            assert!(error.contains("non-stationary"), "{error}");
        }

        // Concave with a gradient far below the resolution: dropping the
        // negative curvature would report a decrease of 4e-9 and certify this
        // local MAXIMUM. The exact model reports 8.7 over the same box.
        let mut concave = |kappa: f64| -> Result<(f64, f64, f64), String> {
            let d = kappa - kappa_hat;
            Ok((-1.0e-9 * d - 0.5 * d * d, -1.0e-9 - d, -1.0))
        };
        let error = curvature_profile_ci_from_analytic_score(
            &mut concave,
            kappa_hat,
            -3.0,
            3.0,
            0.95,
            TEST_RESOLUTION,
            f64::INFINITY,
        )
        .expect_err("a concave point is a maximum along kappa, not an optimum");
        assert!(error.contains("non-stationary"), "{error}");
    }

    /// gam#3509: a chart bound ABOVE the Wilks level brackets the endpoint, so
    /// the profile's slope there says nothing about the interval. On
    /// `V = aκ² − bκ⁴` the profile peaks at `κ² = a/2b` and is falling again at
    /// the bounds `±1`, but `V(±1) = a − b` is still above `½z²`; the interval
    /// is the first crossing of `aκ² − bκ⁴ = ½z²`, found THROUGH the falling
    /// bound rather than refused because of it.
    #[test]
    fn a_bound_above_the_wilks_level_brackets_the_crossing_whatever_its_slope_3509() {
        let (a, b) = (8.0_f64, 5.0_f64);
        let level = 0.95;
        let z = gam_geometry::curvature_estimand::wald_half_width(1.0, level)
            .expect("valid normal quantile");
        let half_threshold = 0.5 * z * z;
        // The fixture must actually be the awkward case: the bound is above the
        // level AND the profile is falling there.
        assert!(a - b > half_threshold && 2.0 * a - 4.0 * b < 0.0);
        let mut profile = |kappa: f64| -> Result<(f64, f64, f64), String> {
            let k2 = kappa * kappa;
            Ok((
                a * k2 - b * k2 * k2,
                2.0 * a * kappa - 4.0 * b * k2 * kappa,
                2.0 * a - 12.0 * b * k2,
            ))
        };
        let ci = curvature_profile_ci_from_analytic_score(
            &mut profile,
            0.0,
            -1.0,
            1.0,
            level,
            TEST_RESOLUTION,
            f64::INFINITY,
        )
        .expect("a bound above the level is a bracket, not a refusal");
        let first_crossing = ((a - (a * a - 4.0 * b * half_threshold).sqrt()) / (2.0 * b)).sqrt();
        // Same endpoint accounting as the quadratic case: the bracket closes
        // once the residual is within the resolution, i.e. within
        // `resolution / slope` of the crossing.
        let slope =
            2.0 * a * first_crossing - 4.0 * b * first_crossing * first_crossing * first_crossing;
        let endpoint_bound = TEST_RESOLUTION / slope;
        assert!(
            (ci.ci_hi - first_crossing).abs() <= endpoint_bound,
            "{ci:?}"
        );
        assert!(
            (ci.ci_lo + first_crossing).abs() <= endpoint_bound,
            "{ci:?}"
        );
        assert!(!ci.lo_at_bound && !ci.hi_at_bound);
    }

    /// gam#3509 / #1464: a bracketing probe OUTSIDE the likelihood set closes the
    /// bracket whatever the profile does beyond it. `V = aκ² − bκ⁴` on
    /// `[−1.2, 1.2]` with `a = 6`, `b = 3` rises through the level at `κ ≈ 0.633`
    /// and turns over at `κ = 1`, so the probe the walk takes at `κ = 0.9` is
    /// outside the set, on a concave stretch whose quadratic model predicts a
    /// fall over the rest of the bracket. That fall is past the crossing and is
    /// not a reason to refuse; the #3245 resolution rewrite had moved the
    /// direction-change refusal ahead of the inside/outside test and refused it.
    #[test]
    fn a_probe_outside_the_level_closes_the_bracket_whatever_follows_it_1464() {
        let (a, b) = (6.0_f64, 3.0_f64);
        let level = 0.95;
        let z = gam_geometry::curvature_estimand::wald_half_width(1.0, level)
            .expect("valid normal quantile");
        let half_threshold = 0.5 * z * z;
        let value = |kappa: f64| a * kappa * kappa - b * kappa.powi(4);
        // The fixture must be the awkward case: the chart bound is above the
        // level, and at the probe κ = 0.9 the profile is outside the set with an
        // outward model that falls by more than the resolution before the bound.
        assert!(value(1.2) > half_threshold && value(0.9) > half_threshold);
        let score = 2.0 * a * 0.9 - 4.0 * b * 0.9_f64.powi(3);
        let curvature = 2.0 * a - 12.0 * b * 0.81;
        assert!(super::profile_model_decrease(score, curvature, 0.0, 0.3) > TEST_RESOLUTION);
        let mut profile = |kappa: f64| -> Result<(f64, f64, f64), String> {
            Ok((
                value(kappa),
                2.0 * a * kappa - 4.0 * b * kappa.powi(3),
                2.0 * a - 12.0 * b * kappa * kappa,
            ))
        };
        let ci = curvature_profile_ci_from_analytic_score(
            &mut profile,
            0.0,
            -1.2,
            1.2,
            level,
            TEST_RESOLUTION,
            f64::INFINITY,
        )
        .expect("an outside probe closes the bracket");
        let first_crossing = ((a - (a * a - 4.0 * b * half_threshold).sqrt()) / (2.0 * b)).sqrt();
        let slope = 2.0 * a * first_crossing - 4.0 * b * first_crossing.powi(3);
        let endpoint_bound = TEST_RESOLUTION / slope;
        assert!((ci.ci_hi - first_crossing).abs() <= endpoint_bound, "{ci:?}");
        assert!((ci.ci_lo + first_crossing).abs() <= endpoint_bound, "{ci:?}");
        assert!(!ci.lo_at_bound && !ci.hi_at_bound);
    }

    /// gam#3509, the other half: a profile whose supremum over the chart is
    /// already inside the Wilks level. `V = aκ² − bκ⁴` with peak `a²/4b` below
    /// `½z²` keeps every κ in the set. The κ-free ceiling certifies that
    /// directly — the interval is the chart, open at both walls, decided
    /// WITHOUT evaluating the profile anywhere but κ̂.
    #[test]
    fn a_ceiling_within_the_wilks_level_certifies_the_whole_chart_3509() {
        let (a, b) = (0.5_f64, 0.3_f64);
        let level = 0.95;
        let z = gam_geometry::curvature_estimand::wald_half_width(1.0, level)
            .expect("valid normal quantile");
        let supremum = a * a / (4.0 * b);
        assert!(supremum < 0.5 * z * z && 2.0 * a - 4.0 * b < 0.0);
        let evaluations = std::cell::Cell::new(0_usize);
        let mut profile = |kappa: f64| -> Result<(f64, f64, f64), String> {
            evaluations.set(evaluations.get() + 1);
            let k2 = kappa * kappa;
            Ok((
                a * k2 - b * k2 * k2,
                2.0 * a * kappa - 4.0 * b * k2 * kappa,
                2.0 * a - 12.0 * b * k2,
            ))
        };
        let ci = curvature_profile_ci_from_analytic_score(
            &mut profile,
            0.0,
            -1.0,
            1.0,
            level,
            TEST_RESOLUTION,
            supremum,
        )
        .expect("the ceiling certifies the whole chart");
        assert_eq!((ci.ci_lo, ci.ci_hi), (-1.0, 1.0));
        assert!(ci.lo_at_bound && ci.hi_at_bound);
        assert_eq!(
            ci.verdict,
            gam_geometry::curvature_estimand::CurvatureVerdict::Flat
        );
        // κ̂ only. A single evaluation is what distinguishes the certificate
        // from an endpoint search that happened to walk out to the same walls.
        assert_eq!(
            evaluations.get(),
            1,
            "the ceiling must settle the interval without an endpoint search"
        );

        // A ceiling below V_p(κ̂) contradicts the profile it claims to bound,
        // so the fit is refused rather than reported.
        let mut shifted = |kappa: f64| -> Result<(f64, f64, f64), String> {
            let k2 = kappa * kappa;
            Ok((
                1.0 + a * k2 - b * k2 * k2,
                2.0 * a * kappa - 4.0 * b * k2 * kappa,
                2.0 * a - 12.0 * b * k2,
            ))
        };
        assert!(
            curvature_profile_ci_from_analytic_score(
                &mut shifted,
                0.0,
                -1.0,
                1.0,
                level,
                TEST_RESOLUTION,
                supremum,
            )
            .is_err(),
            "a ceiling below V(kappa_hat) is a violated premise, not a usable bound"
        );
    }
}

#[cfg(test)]
mod nfree_gate_tests {
    use super::nfree_skip_gate_status_from_parts;

    #[test]
    fn value_only_nfree_gate_does_not_require_basis_skip_witness() {
        let gate = nfree_skip_gate_status_from_parts(
            true,  // shape
            true,  // Chebyshev Gram value covers this ψ
            false, // reduced-basis skip witness absent across a rotation seam
            false, // gradient coverage irrelevant for a value-only cost probe
            true,  // penalty can be re-keyed without rows
            true,  // design revision is pinned
            false, // no Hessian request
            false, // value-only cost probe
        );
        assert!(
            gate.would_skip(false),
            "value-only κ cost probes must stay n-free when the Gram value is certified; \
             the reduced-basis skip witness is required only for beta/gradient probes"
        );
    }

    #[test]
    fn gradient_nfree_gate_still_requires_basis_skip_witness() {
        let gate =
            nfree_skip_gate_status_from_parts(true, true, false, true, true, true, false, true);
        assert!(
            !gate.would_skip(true),
            "gradient probes return beta/gradient objects in a reduced basis and must not \
             skip the row lane without the reduced-basis witness"
        );
    }
}
