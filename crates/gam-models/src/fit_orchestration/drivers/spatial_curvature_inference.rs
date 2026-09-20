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

/// Is the profile's remaining first-order decrease at a point RESOLVABLE?
///
/// This is the outer solve's own stationarity measure (#2458, #2954) restricted
/// to the one κ coordinate: the curvature-denominated Newton decrement
/// `λ² = g²/H` against the criterion's statistical resolution
/// `τ_stat = 1/(2n)` ([`gam_solve::rho_optimizer::outer_statistical_resolution`]).
/// A decrease the quadratic model puts below `τ_stat` moves no reported
/// quantity by more than `n^{-1/2}` standard errors, so the point is certified;
/// one above it is a real descent the estimate or the bracket has missed. With
/// `H ≤ 0` the local model has no bounded minimum and any nonzero gradient is a
/// resolvable descent — the outer certificate does not certify an indefinite
/// point either. `g = 0` certifies at any curvature.
///
/// Before #3453 the inference layer re-judged κ̂ against an independent bar
/// `max(tol, √ε)·(1 + |V_p|)`: not derived, scaled by the ADDITIVE level of
/// the criterion (a constant in `V_p` moved it), and in raw gradient units, so
/// a κ̂ the outer solve had certified at `λ² ≈ 3e-7 ≪ τ_stat` was refused
/// because `|g| = 4e-3` exceeded `9e-5`.
fn curvature_profile_descent_is_resolvable(gradient: f64, curvature: f64, tau_stat: f64) -> bool {
    if gradient == 0.0 {
        return false;
    }
    !(curvature > 0.0) || gradient * gradient / curvature > tau_stat
}

/// Solve one Wilks endpoint of the analytic curvature profile between κ̂ and
/// the chart bound on that side.
///
/// Each CI endpoint solves the likelihood-ratio equation directly inside the
/// chart-bound bracket with safeguarded Newton steps; bisection is the
/// guaranteed-progress fallback. A bound is reported as open only when the
/// analytic score certifies that the connected likelihood set containing κ̂
/// remains monotone all the way to that bound. Every judgement is made at the
/// criterion's statistical resolution `tau_stat`, the bar the outer solve
/// certified κ̂ with: a direction change is refused only when its decrement is
/// resolvable, a bound refutes κ̂ as the minimum only when it undercuts
/// `V_p(κ̂)` by more than `tau_stat`, and the crossing is resolved once its
/// likelihood-ratio residual is below `tau_stat` — an endpoint whose `½LR`
/// misses the Wilks threshold by less than the criterion's resolution is not
/// distinguishable from the exact one.
fn curvature_profile_lr_endpoint<F>(
    profile: &mut F,
    kappa_hat: f64,
    value_hat: f64,
    bound: f64,
    half_threshold: f64,
    tau_stat: f64,
) -> Result<(f64, bool), String>
where
    F: FnMut(f64) -> Result<(f64, f64, f64), String>,
{
    let direction = (bound - kappa_hat).signum();
    if direction == 0.0 {
        return Ok((bound, true));
    }

    let (bound_value, bound_score, bound_curvature) = profile(bound)?;
    let outward_score = direction * bound_score;
    if outward_score < 0.0
        && curvature_profile_descent_is_resolvable(bound_score, bound_curvature, tau_stat)
    {
        return Err(format!(
            "curvature profile is not outward-monotone at chart bound {bound}: \
             outward score {outward_score:.6e} at curvature {bound_curvature:.6e} is a \
             resolvable direction change (tau_stat={tau_stat:.6e})"
        ));
    }
    if bound_value < value_hat - tau_stat {
        return Err(format!(
            "fitted curvature is not the minimum of its inference profile: \
             V(bound={bound})={bound_value:.6e} < V(kappa_hat)={value_hat:.6e} \
             by more than tau_stat={tau_stat:.6e}"
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
    let mut inside_x = kappa_hat;
    let mut outside_x = bound;
    let mut outside_residual = bound_residual;
    let mut outside_score = bound_score;
    while outside_residual > tau_stat {
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
        let outward_score = direction * score;
        if outward_score < 0.0
            && curvature_profile_descent_is_resolvable(score, curvature, tau_stat)
        {
            return Err(format!(
                "curvature profile changed direction before its likelihood crossing at \
                 kappa={probe}: outward score {outward_score:.6e} at curvature \
                 {curvature:.6e} is a resolvable direction change (tau_stat={tau_stat:.6e})"
            ));
        }
        let residual = value - value_hat - half_threshold;
        if residual >= 0.0 {
            outside_x = probe;
            outside_residual = residual;
            outside_score = score;
        } else {
            inside_x = probe;
        }
    }
    // `outside_x` already holds the analytic score and residual evaluated there,
    // so one final Newton step costs no additional profile evaluation and
    // squares the resolved residual. It is taken only when it lands inside the
    // certified bracket; otherwise the midpoint stands (reached only when the
    // bracket collapsed to adjacent floats before the residual resolved).
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

/// The Wilks profile CI for κ from the analytic profile `(V_p, V_p', V_p'')`.
///
/// `tau_stat` is the criterion's statistical resolution over the SAME `n` the
/// κ outer problem declares, so κ̂ is re-examined with the decrement the outer
/// solve certified it with rather than a second bar (#3453), and it is the
/// only tolerance here: rail recognition, stationarity, monotonicity and the
/// endpoint solves are all denominated in it (#3245).
fn curvature_profile_ci_from_analytic_score<F>(
    profile: &mut F,
    kappa_hat: f64,
    kappa_min: f64,
    kappa_max: f64,
    level: f64,
    tau_stat: f64,
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
    if !(tau_stat.is_finite() && tau_stat > 0.0) {
        return Err(format!(
            "curvature profile requires a positive finite statistical resolution, got {tau_stat}"
        ));
    }
    let z = gam_geometry::curvature_estimand::wald_half_width(1.0, level)
        .ok_or_else(|| "curvature profile threshold is not finite".to_string())?;
    let half_threshold = 0.5 * z * z;
    let (value_hat, score_hat, curvature_hat) = profile(kappa_hat)?;
    // At a bound, "stationary" means the score points OUT of the box, not that
    // it vanishes. That is the routine knowing κ̂ is a box readout — and before
    // #2687 it then threw the knowledge away and reported κ̂ as an estimate. It
    // is now carried on the report.
    //
    // κ̂ sits ON a rail when the profile cannot tell the two apart: the local
    // model's largest possible change over the gap, `|g|·gap + ½|H|·gap²`,
    // is within `τ_stat`. A κ̂ the outer solve projected onto the face has
    // `gap = 0` exactly; one a resolvable distance inside is interior.
    let rail_indistinguishable = |gap: f64| {
        score_hat.abs() * gap + 0.5 * curvature_hat.abs() * gap * gap <= tau_stat
    };
    let lower_gap = kappa_hat - kappa_min;
    let upper_gap = kappa_max - kappa_hat;
    let at_lower = lower_gap <= upper_gap && rail_indistinguishable(lower_gap);
    let at_upper = !at_lower && rail_indistinguishable(upper_gap);
    let kappa_hat_support = if at_lower {
        gam_geometry::curvature_estimand::KappaEstimateSupport::RailedAtLowerBound
    } else if at_upper {
        gam_geometry::curvature_estimand::KappaEstimateSupport::RailedAtUpperBound
    } else {
        gam_geometry::curvature_estimand::KappaEstimateSupport::Interior
    };
    // The projected gradient: at a rail only the inward-descending component
    // survives (a score pointing out of the box is the boundary optimum's KKT
    // multiplier, not a missed decrease).
    let projected_score = if at_lower {
        score_hat.min(0.0)
    } else if at_upper {
        score_hat.max(0.0)
    } else {
        score_hat
    };
    if curvature_profile_descent_is_resolvable(projected_score, curvature_hat, tau_stat) {
        // Name what was refused AGAINST, not just that something was refused.
        // A κ̂ that failed this check is either a genuine interior non-optimum or
        // a rail that was not recognised, and the two need opposite repairs — so
        // the message has to carry the box, both gaps, and the classification
        // (#2687).
        let decrement = if curvature_hat > 0.0 {
            projected_score * projected_score / curvature_hat
        } else {
            f64::INFINITY
        };
        return Err(format!(
            "curvature inference rejected a non-stationary point estimate: \
             kappa_hat={kappa_hat}, score={score_hat:.6e}, curvature={curvature_hat:.6e}, \
             decrement={decrement:.6e} > tau_stat={tau_stat:.6e}; \
             box=[{kappa_min}, {kappa_max}], gap_to_lower={lower_gap:.6e}, \
             gap_to_upper={upper_gap:.6e}, classified={}",
            kappa_hat_support.label()
        ));
    }

    let (ci_lo, lo_at_bound) = curvature_profile_lr_endpoint(
        profile,
        kappa_hat,
        value_hat,
        kappa_min,
        half_threshold,
        tau_stat,
    )?;
    let (ci_hi, hi_at_bound) = curvature_profile_lr_endpoint(
        profile,
        kappa_hat,
        value_hat,
        kappa_max,
        half_threshold,
        tau_stat,
    )?;
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

/// Compute the #944 curvature inference for the constant-curvature smooth at
/// `term_idx`, given the already-fitted resolved spec (carrying κ̂) and the same
/// fit inputs used to produce it.
///
/// The point estimate and inference share the same continuously smoothing-
/// profiled Gaussian REML evidence and its analytic profile score and
/// curvature; κ̂ is re-examined with the outer solve's own decrement
/// certificate at `τ_stat = 1/(2n)`.
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
    validate_constant_curvature_profile_inputs(weights, offset, &family)?;
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

    // CI and flatness revisit κ̂ and κ=0. The shared profile caches each joint
    // value/analytic-score pair so every statistic consumes the same evaluation.
    let mut v_p = |kappa: f64| -> Result<(f64, f64, f64), String> {
        if !kappa.is_finite() {
            return Err(format!("V_p probed a non-finite κ = {kappa}"));
        }
        profile.evaluate(kappa).map_err(|error| {
            format!("analytic curvature profile at kappa={kappa} failed: {error}")
        })
    };
    // The κ outer problem declares `n_obs = y.len()` (see
    // `constant_curvature_kappa_problem`), so this is the resolution its
    // decrement certificate accepted κ̂ at.
    let tau_stat = gam_solve::rho_optimizer::outer_statistical_resolution(y.len())
        .ok_or_else(|| {
            EstimationError::InvalidInput(
                "curvature inference requires at least one observation".to_string(),
            )
        })?;
    let ci = curvature_profile_ci_from_analytic_score(
        &mut v_p,
        kappa_hat,
        kappa_min,
        kappa_max,
        level,
        tau_stat,
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

    /// `τ_stat = 1/(2n)` for the `n = 200` rows of the #3453 reproducers.
    const TAU_STAT_N200: f64 = 0.5 / 200.0;

    #[test]
    fn analytic_profile_score_finds_exact_quadratic_lr_crossings() {
        let kappa_hat = -0.37;
        let curvature = 16.0;
        let level = 0.95;
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
            TAU_STAT_N200,
        )
        .expect("analytic quadratic profile CI");
        let z = gam_geometry::curvature_estimand::wald_half_width(1.0, level)
            .expect("valid normal quantile");
        let expected_half_width = z / curvature.sqrt();
        // Each endpoint is resolved once its Wilks residual `V − V̂ − z²/2` is
        // within `τ_stat`. The profile is convex, so it lies above its tangent at
        // the exact crossing, whose slope is `s⋆ = H·w = z·√H`: a residual
        // `r ≤ τ_stat` therefore places the endpoint within `τ_stat / s⋆` of the
        // exact one in κ. That is the whole accuracy the criterion can resolve.
        let crossing_slope = curvature * expected_half_width;
        let kappa_resolution = TAU_STAT_N200 / crossing_slope;
        for (endpoint, exact) in [
            (ci.ci_lo, kappa_hat - expected_half_width),
            (ci.ci_hi, kappa_hat + expected_half_width),
        ] {
            let d = endpoint - kappa_hat;
            let residual = 0.5 * curvature * d * d - 0.5 * z * z;
            assert!(residual.abs() <= TAU_STAT_N200, "Wilks residual {residual:.3e}");
            assert!(
                (endpoint - exact).abs() <= kappa_resolution,
                "endpoint {endpoint} vs exact {exact} (resolution {kappa_resolution:.3e})"
            );
        }
        assert!(!ci.lo_at_bound && !ci.hi_at_bound);
    }

    #[test]
    fn analytic_profile_marks_chart_bound_when_wilks_set_never_crosses() {
        let mut profile =
            |kappa: f64| -> Result<(f64, f64, f64), String> { Ok((0.5 * kappa * kappa, kappa, 1.0)) };
        let ci = curvature_profile_ci_from_analytic_score(
            &mut profile,
            0.0,
            -0.1,
            0.1,
            0.95,
            TAU_STAT_N200,
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
            TAU_STAT_N200,
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
            TAU_STAT_N200,
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
                TAU_STAT_N200,
            )
            .is_err(),
            "a non-stationary INTERIOR point is not an optimum and must still be refused"
        );
    }

    /// gam#3453: κ̂ is re-examined with the outer solve's own decrement
    /// `λ² = g²/H` against `τ_stat`, not a second bar in raw gradient units.
    /// The first profile is the planted-hyperbolic reproducer's point
    /// (`g = −4.04e-3`, `V_p'' ≈ 59`, so `λ² ≈ 2.8e-7 ≪ τ_stat`): the outer
    /// solve certified it, and the old `max(tol, √ε)·(1 + |V_p|)` bar
    /// (`≈ 9e-5` there) refused it. Its acceptance must not depend on the
    /// additive level of `V_p`, which that bar scaled with.
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
        let certified_gradient = -4.04e-3;
        assert!(certified_gradient * certified_gradient / curvature < TAU_STAT_N200);
        for level in [0.0, 5.8e2, 1.0e7] {
            let mut profile = quadratic(level, certified_gradient);
            let ci = curvature_profile_ci_from_analytic_score(
                &mut profile,
                kappa_hat,
                -3.0,
                3.0,
                0.95,
                TAU_STAT_N200,
            )
            .unwrap_or_else(|e| panic!("certified κ̂ refused at V level {level}: {e}"));
            assert_eq!(
                ci.kappa_hat_support,
                gam_geometry::curvature_estimand::KappaEstimateSupport::Interior
            );
            assert!(ci.ci_lo < kappa_hat && kappa_hat < ci.ci_hi && ci.ci_hi < 0.0);
        }
        // A gradient whose decrement exceeds τ_stat is a real missed descent and
        // is still refused, at every level.
        let resolvable_gradient = 1.1 * (TAU_STAT_N200 * curvature).sqrt();
        for level in [0.0, 1.0e7] {
            let mut profile = quadratic(level, resolvable_gradient);
            let error = curvature_profile_ci_from_analytic_score(
                &mut profile,
                kappa_hat,
                -3.0,
                3.0,
                0.95,
                TAU_STAT_N200,
            )
            .expect_err("a resolvable decrement is not a stationary point");
            assert!(error.contains("non-stationary"), "{error}");
        }
        // Concave with any nonzero gradient: no bounded quadratic minimum, so
        // not certified however small the gradient.
        let mut concave = |kappa: f64| -> Result<(f64, f64, f64), String> {
            let d = kappa - kappa_hat;
            Ok((-1.0e-9 * d - 0.5 * d * d, -1.0e-9 - d, -1.0))
        };
        assert!(
            curvature_profile_ci_from_analytic_score(
                &mut concave,
                kappa_hat,
                -3.0,
                3.0,
                0.95,
                TAU_STAT_N200,
            )
            .is_err()
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
