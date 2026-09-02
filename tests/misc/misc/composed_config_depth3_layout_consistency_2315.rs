//! Gap 1 of #2315: a single GENERALIZED "2-of-everything, depth-3"
//! composed-configuration standing harness.
//!
//! The pre-existing per-site regressions (`composed_score_link_influence_rho_
//! layout_advances_by_emitted_counts_2315` in `bms/block_specs.rs`,
//! `multi_block_group_priors_follow_realized_penalty_order_2315` in
//! `coefficient_groups.rs`, and the penalty-label regressions in
//! `penalty_labels.rs`) each pin ONE composition site. None of them build a
//! model that stacks two of every composable ingredient at once and asserts the
//! whole realized layout stays self-consistent. That "silent-wrong-answer under
//! composition" class is exactly what shipped as #2287–#2292: a penalty piece
//! that received the right-looking slice but the wrong optimizer coordinate.
//!
//! This harness drives the fully-public production composition entry point
//!     gam::families::custom_family::realize_coefficient_groups_for_custom_family
//! which is the layout builder the fit itself calls. Internally it fans out to
//! the exact helpers the per-site regressions cover:
//!   * `validate_blockspecs` / `validate_blockspec_consistency` (block topology),
//!   * `penalty_label_layout_with_joint` (physical→outer coordinate law),
//!   * `resolved_physical_penalty_label` (the `__block_i_penalty_j` convention).
//! Coefficient groups are the public composable "influence / score-link layer"
//! analogue: each is a separate independent Gaussian prior factor layered onto
//! the base smooth prior, exactly like the BMS score_warp / link_dev / influence
//! absorbers whose `push_deviation_aux_blockspecs` sibling is `pub(crate)`.
//!
//! Every zoo entry is a full 2-of-everything, depth-3 configuration:
//!   * >= 2 parameter blocks (smooth terms),
//!   * >= 2 blocks carrying >= 2 base penalties,
//!   * >= 2 coefficient groups (composable prior-factor layers),
//!   * a coefficient-group hierarchy of nesting depth >= 3.
//! For each we re-derive the emitted penalty labels, the optimizer-coordinate
//! (outer) labels, and the per-coordinate priors independently from the realized
//! block specs and assert they match production coordinate-for-coordinate.

