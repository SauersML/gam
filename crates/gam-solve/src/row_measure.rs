//! Row-subsample mask handle for trust-region invariant enforcement.
//!
//! The `RowSubsampleMask` data type and its pure data methods (`full_data`,
//! `subsample`, `indices_and_weights`) have descended to `gam-problem` so
//! lower tiers can consume the row measure without depending on `gam-solve`;
//! they are re-exported here unchanged.
//!
//! The `BlockwiseFitOptions`-coupled constructor stays here: it reads the
//! outer optimizer's `outer_score_subsample` (an option type that lives above
//! `gam-problem`), so it cannot descend. Because the data type is now foreign
//! to this crate, the constructor is provided through the [`RowSubsampleMaskExt`]
//! extension trait rather than an inherent method.

use std::sync::Arc;

use gam_model_api::families::custom_family::BlockwiseFitOptions;

pub use gam_problem::RowSubsampleMask;

/// Extension trait carrying the `BlockwiseFitOptions`-coupled constructor for
/// [`RowSubsampleMask`]. The data type lives in `gam-problem`; this constructor
/// stays in `gam-solve` because it depends on `BlockwiseFitOptions`.
pub trait RowSubsampleMaskExt {
    /// Build a `RowSubsampleMask` from blockwise-fit options. The outer
    /// optimizer is the sole source of `outer_score_subsample`; inner
    /// paths read this once at the top of each TR iteration and freeze
    /// it for every quantity in that iteration.
    fn from_options(options: &BlockwiseFitOptions, n: usize) -> Self;
}

impl RowSubsampleMaskExt for RowSubsampleMask {
    fn from_options(options: &BlockwiseFitOptions, n: usize) -> Self {
        match options.outer_score_subsample.as_ref() {
            Some(mask) => Self::subsample(Arc::clone(mask)),
            None => Self::full_data(n),
        }
    }
}
