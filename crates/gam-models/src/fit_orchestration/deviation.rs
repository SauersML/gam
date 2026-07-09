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
    logslope_linkwiggle: Option<&LinkWiggleFormulaSpec>,
) -> Result<MarginalSlopeDeviationRouting, String> {
    Ok(MarginalSlopeDeviationRouting {
        score_warp: logslope_linkwiggle
            .map(deviation_block_config_from_formula_linkwiggle)
            .transpose()?,
        link_dev: main_linkwiggle
            .map(deviation_block_config_from_formula_linkwiggle)
            .transpose()?,
    })
}
