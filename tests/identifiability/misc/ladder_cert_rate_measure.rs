//! #979 empirics: measure the non-affine GL-ladder certification distribution
//! on the REAL flex marginal-slope path (score_warp + link_dev → non-affine
//! denested cells), instead of assuming "narrow knot cells certify early".
//!
//! The ladder walks 12→24→48→96→192 nodes and accepts the first two-rule
//! agreement, falling through to a terminal 384-node rule otherwise. It is a
//! win when cells certify at a low rung (≪384 nodes) and a ~2× regression on
//! cells that fall through to 384. This test fits a flex problem and prints
//! the per-rung histogram so the ladder's real cost is data, not faith.

