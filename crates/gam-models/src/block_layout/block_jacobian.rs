//! Shared additive-plus-wiggle block-Jacobian dispatcher.
//!
//! Several multi-output custom families (survival location-scale, the Binomial
//! location-scale-wiggle family, …) build the [`BlockEffectiveJacobian`] for one
//! parameter block with the identical structure:
//!
//! * a set of **additive blocks**, each of which drives exactly one family
//!   output (`own_output == block_idx`) via its effective design matrix; and
//! * an optional **wiggle / link block**, which modulates the inverse link
//!   nonlinearly and therefore contributes an all-zero effective linear
//!   Jacobian of shape `(n × p_wiggle)` (anchored at `own_output = 0`).
//!
//! Only the family name, the number of family outputs, the additive block ids
//! and the wiggle block id vary between callers. This module holds the single
//! canonical implementation so the dispatch is not re-typed by hand.

/// Static description of an additive-plus-wiggle family's block layout, used to
/// build the per-block [`BlockEffectiveJacobian`] in one place.
///
/// `additive_blocks` lists the block ids that contribute a linear additive term
/// to a single family output; each such block drives output `block_idx`
/// (i.e. `own_output == block_idx`). `wiggle_block` is the optional nonlinear
/// link-modulation block whose effective linear Jacobian is all zeros; its row
/// count is taken from the first additive block's design.
pub struct AdditiveWiggleBlockLayout<'a> {
    /// Family name used as the message prefix and `effective_design` context,
    /// e.g. `"SurvivalLocationScaleFamily"`.
    pub family: &'a str,
    /// Total number of stacked family outputs (e.g. 3 for survival
    /// location-scale, 2 for Binomial location-scale).
    pub n_outputs: usize,
    /// Block ids that contribute a linear additive term to their own output.
    pub additive_blocks: &'a [usize],
    /// Optional nonlinear link-modulation block id (all-zero linear Jacobian).
    pub wiggle_block: Option<usize>,
}

