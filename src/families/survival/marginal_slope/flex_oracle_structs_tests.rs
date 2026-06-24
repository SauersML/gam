//! #932-2 cutover: return shapes of the test-only hand directional/bidirectional
//! timepoint oracle producers (`timepoint_exact::{directional,bidirectional}_oracle_tests`).
//!
//! The production contracted path (`timepoint_exact::contracted`) reads the Block-10
//! directional/bidirectional packs directly from the single-source `Jet3`/`Jet4`
//! builders, so these structs are no longer production surface — they live here, in a
//! test-masked module, consumed only by the hand oracle (which pins the jet path via
//! the `flex_jet` `flex_timepoint_inputs_jet{3,4}_*_matches_hand_932` gates) and the
//! `tests.rs` finite-difference witnesses.

use ndarray::{Array1, Array2};

/// Directional extensions of a timepoint's exact quantities, contracted with a
/// single direction — the pieces the hand third-order NLL contraction oracle composes.
pub(crate) struct SurvivalFlexTimepointDirectionalExact {
    pub(crate) eta_uv_dir: Array2<f64>,
    pub(crate) eta_u_dir: Array1<f64>,
    pub(crate) chi_u_dir: Array1<f64>,
    pub(crate) chi_uv_dir: Array2<f64>,
    pub(crate) d_u_dir: Array1<f64>,
    pub(crate) d_uv_dir: Array2<f64>,
    pub(crate) a_uv_dir: Array2<f64>,
}

/// Mixed second-directional extensions `D_{d1} D_{d2}` of a timepoint's exact
/// quantities — the hand fourth-order NLL contraction oracle's bidirectional pack.
pub(crate) struct SurvivalFlexTimepointBiDirectionalExact {
    pub(crate) eta_uv_uv: Array2<f64>,
    pub(crate) chi_uv_uv: Array2<f64>,
    pub(crate) d_uv_uv: Array2<f64>,
}
