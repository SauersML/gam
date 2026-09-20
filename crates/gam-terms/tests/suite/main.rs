//! Integration-test harness for gam-terms: every module here was a
//! standalone tests/*.rs crate and therefore its own link of gam-terms and
//! its dependency tree. One binary, same tests, same names.

mod average_derivative_design;
mod duchon_lazy_anisotropic_reparam_1818;
mod duchon_single_design_build_1718;
mod factor_by_level_null_ridge_replays_the_fit;
mod knot_selection_is_scale_equivariant_2750;
mod latent_collision_design_gradient;
mod latent_coord_design_jacobian_frame_fd_2643;
mod matern_aniso_structural_ridge_derivative_fd;
mod measure_jet_affine_null_survives_centering_2751;
mod measure_jet_psi_box_is_its_own_geometry_2750;
mod measure_jet_range_moves_the_span_2761;
mod parametric_orthogonality_costs_no_dimension_2747;
mod pca_term_freezes_for_persistence_2627;
mod probe_2761_penalty_topology;
