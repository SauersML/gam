//! Canonical routing from formula-level link-wiggle declarations to the
//! marginal-slope deviation blocks consumed by the model families.
//!
//! Formula materialization and every application frontend use this module.
//! Keeping the cubic-runtime constraint and penalty defaults here prevents the
//! CLI and library fit paths from accepting different models.

use super::*;

fn deviation_block_config_from_formula_linkwiggle(
    wiggle: &LinkWiggleFormulaSpec,
) -> Result<DeviationBlockConfig, String> {
    // The score-warp / link-deviation runtime is a cubic I-spline: its span
    // tables, C2-continuous construction, and derivative operators are all
    // structurally cubic. The formula parser remains general because other
    // wiggle consumers support arbitrary degrees, so enforce this constraint
    // at the routing boundary shared by every frontend.
    if wiggle.degree != 3 {
        return Err(format!(
            "linkwiggle() degree must be 3 when routed into the score-warp / \
             link-deviation block: that runtime is a cubic I-spline and only \
             supports cubic splines; got degree={}",
            wiggle.degree
        ));
    }
    let defaults = WigglePenaltyConfig::cubic_triple_operator_default();
    Ok(DeviationBlockConfig {
        degree: wiggle.degree,
        num_internal_knots: wiggle.num_internal_knots,
        penalty_order: *wiggle.penalty_orders.iter().max().unwrap_or(&2),
        penalty_orders: wiggle.penalty_orders.clone(),
        double_penalty: wiggle.double_penalty,
        monotonicity_eps: defaults.monotonicity_eps,
    })
}

#[derive(Debug)]
pub struct MarginalSlopeDeviationRouting {
    pub score_warp: Option<DeviationBlockConfig>,
    pub link_dev: Option<DeviationBlockConfig>,
}

pub fn route_marginal_slope_deviation_blocks(
    main_linkwiggle: Option<&LinkWiggleFormulaSpec>,
    slope_linkwiggle: Option<&LinkWiggleFormulaSpec>,
) -> Result<MarginalSlopeDeviationRouting, String> {
    Ok(MarginalSlopeDeviationRouting {
        score_warp: slope_linkwiggle
            .map(deviation_block_config_from_formula_linkwiggle)
            .transpose()?,
        link_dev: main_linkwiggle
            .map(deviation_block_config_from_formula_linkwiggle)
            .transpose()?,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The `linkwiggle()` default, as the formula DSL mints it.
    fn default_linkwiggle() -> LinkWiggleFormulaSpec {
        let defaults = WigglePenaltyConfig::cubic_triple_operator_default();
        LinkWiggleFormulaSpec {
            degree: defaults.degree,
            num_internal_knots: defaults.num_internal_knots,
            penalty_orders: defaults.penalty_orders,
            double_penalty: defaults.double_penalty,
        }
    }

    /// gam#4564 — `linkwiggle()` in BOTH formulas activates BOTH flex blocks.
    ///
    /// This is the premise every cost statement about that fit rests on, and
    /// nothing pinned it. The BMS flex row program's primary width is
    /// `r = 2 + dim(score warp) + dim(link deviation)`, so which formula
    /// activates which block decides whether a model reaches `r = 2` (the rigid
    /// specialization every wiggle-free arm runs) or the widest configuration
    /// the product ships. The routing is the only place that decision is made:
    /// the SLOPE formula's wiggle becomes the score warp and the MEAN formula's
    /// becomes the link deviation, so the shipped arm of gam#4564 — which
    /// spells `linkwiggle()` in each — carries both.
    ///
    /// The three one-sided cases are asserted beside it, because "both are
    /// `Some`" is only informative if a single wiggle does NOT produce both.
    #[test]
    fn linkwiggle_in_both_formulas_routes_both_flex_blocks_4564() {
        let wiggle = default_linkwiggle();

        let neither = route_marginal_slope_deviation_blocks(None, None)
            .expect("no wiggle routes no deviation block");
        assert!(
            neither.score_warp.is_none() && neither.link_dev.is_none(),
            "no linkwiggle() must leave the rigid two-dimensional row"
        );

        let slope_only = route_marginal_slope_deviation_blocks(None, Some(&wiggle))
            .expect("a slope-formula wiggle routes");
        assert!(
            slope_only.score_warp.is_some() && slope_only.link_dev.is_none(),
            "the slope formula's linkwiggle() is the score warp, and only that"
        );

        let main_only = route_marginal_slope_deviation_blocks(Some(&wiggle), None)
            .expect("a mean-formula wiggle routes");
        assert!(
            main_only.link_dev.is_some() && main_only.score_warp.is_none(),
            "the mean formula's linkwiggle() is the link deviation, and only that"
        );

        let both = route_marginal_slope_deviation_blocks(Some(&wiggle), Some(&wiggle))
            .expect("a wiggle in each formula routes");
        let score_warp = both
            .score_warp
            .as_ref()
            .expect("the slope formula's wiggle is the score warp");
        let link_dev = both
            .link_dev
            .as_ref()
            .expect("the mean formula's wiggle is the link deviation");

        // Each block carries the same default, and that default is what fixes
        // both halves of the cost: the knot count sets the block's basis
        // dimension and so the primary width, and the penalty count sets how
        // many smoothing parameters the outer search must move.
        for (label, cfg) in [("score warp", score_warp), ("link deviation", link_dev)] {
            assert_eq!(cfg.degree, 3, "{label}: the deviation runtime is cubic");
            assert_eq!(
                cfg.num_internal_knots, 8,
                "{label}: the linkwiggle() default is eight internal knots"
            );
            assert_eq!(
                cfg.penalty_orders,
                vec![1, 2, 3],
                "{label}: the default penalises slope, curvature and curvature change"
            );
            assert!(
                cfg.double_penalty,
                "{label}: the default carries its own double penalty"
            );
            assert_eq!(
                cfg.penalty_orders.len() + usize::from(cfg.double_penalty),
                4,
                "{label}: four smoothing parameters per default linkwiggle() block"
            );
        }
    }
}
